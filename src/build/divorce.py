"""Divorce-property setting: rebuild the claim sample (paper Appendix C.4).

Source: annual full-text archives of China Judgments Online decisions distributed by the
third-party ``cncases/cases`` project (2009-2018 and 2021 archives). See
``data/download/divorce.md``. ``raw_dir`` is a folder holding the eleven annual ZIP files
(``2009年裁判文书数据.zip`` ... ``2021年裁判文书数据.zip``; any sub-folder is searched).

The construction runs in five stages (counts printed by
``python -m src.build.divorce <raw_dir>``):

1. Divorce-related rows. Every CSV member of every annual ZIP is read to the end; a row is
   kept when its case name or cause contains 离婚, 夫妻财产 or 婚内财产.
   50,674,220 rows read -> 3,321,969 divorce-related rows. The kept rows are cached as
   ``<cache>/<year>.jsonl.gz`` so later stages do not re-read the 100 GB archives.
2. Court-docket deduplication. Rows sharing the same court and docket number are one
   record; the version with the longest normalized full text is kept (ties: larger text
   SHA-256). Rows without court or docket are keyed by their text hash. -> 3,162,013 records.
3. Case screening, claim extraction and claim-level outcome (rules in ``extract_case``,
   ``extract_claims`` and ``label_claim``). A case must be a first-instance civil judgment
   with a recoverable current disposition; a separate post-divorce property action or a
   judgment that grants the divorce; exactly one explicitly identified male and one female
   spouse in the party headers (no name-based sex inference); substantive plaintiff and
   defendant statements; no third party or counterclaim; no bride-price/dowry term. Each
   current plaintiff property prayer with linked defendant opposition is a candidate claim;
   it is retained only if the court fully granted (``FULLY_UPHELD``) or fully denied
   (``DENIED``) it. -> 38,333 eligible cases, 34,151 candidate claims, 3,022 fully granted
   or denied claims in 2,601 cases.
4. Prompt-layer screen. Party statements are cut at the first sentence-boundary transition
   into the court's own findings or the evidence listing (``clean_statement``); a claim is
   dropped when it is an automatically split sub-claim, when its target is not in the cleaned
   plaintiff statement, or when the linked opposition is not in the cleaned defendant
   statement. Two targets that are not property prayers are excluded by ID, and for ten
   claims a reviewed opposing phrase (identified by its SHA-256, no text is shipped) stands
   in for an extractor span that pointed past the defendant statement. -> 2,610 candidate
   records in 2,362 cases.
5. Source-based reconstruction (``reconstruct``). Each candidate's current prayer and both
   parties' statements, evidence lists and cross-examination are rebuilt from exact spans of
   the normalized judgment; court findings, reasoning and disposition never enter the input.
   A record is retained only if the reconstruction passes every automatic check; otherwise
   it goes to the review queue and is excluded. Fragments that rebuild to the same prayer are
   merged, keeping a retained fragment first and then the smallest source claim ID.
   -> 1,826 retained claims in 1,696 cases, 774 review-queue records, 10 merged duplicate
   fragments. A whole-order dismissal of all current requests is re-applied to the rebuilt
   prayer when the target's scope changed, so such records are not queued for that reason.

Claim-level outcomes are extracted from the court's reasoning and disposition by the rules
below.

The real outcome is the court's claim-level ruling mapped to the spouses through the
plaintiff's sex: ``FULLY_UPHELD`` favors the plaintiff and ``DENIED`` the defendant, so
``real_underdog = 1`` exactly when the favored party is the female spouse. Unit = claim
(``unit_id``), cluster = source case (``cluster_id``), weight 1.

Judgment texts contain personal information; the cache files and anything returned by
``load_records`` must stay local and must not be redistributed.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import html
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

YEARS = (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2021)

# SHA-256 and size of each annual archive as used for the paper.
ARCHIVES = {
    2009: ('b8b1e1d4c3be4f19a33323c2a4c537fb409e32f3a3e101cdb0281c799b681527', 84726043),
    2010: ('c6507341f3651b828f96bfaa156342be317edb24bad5400ba21d6f23a40344b1', 212355227),
    2011: ('d8aedeba761b83242a8bd3e94f1cabfb7e6283f64a5481ef12bca6d0c375e447', 128542106),
    2012: ('08d6aacd0126701bef240cd221d02897677cde0f82aab243b7c81fc7faa0eb71', 218429588),
    2013: ('10925fa4104694d7665747f320278b6420a5956efd8fe96db247d622bc65ad49', 1543026562),
    2014: ('bf71394ed41c8813c268ba5c7c48f7b2c136818c6671a74002b9d846a8a0225c', 7150687964),
    2015: ('2a8d16882981bc494a83c34a71d9a38fe301a1d6bd9f58e4963187532f25b665', 10073743020),
    2016: ('1684a3f5664b85ffffc27af44392e2372ee01257e8c47144f6755277590bff31', 11935695545),
    2017: ('1c588f260f96549c26ef521ed642382a0145422e3a130ce3b29db6005265b92e', 13292189254),
    2018: ('9c9374572d73a7b8747f47b1a12316c22ba4c4269eac046551035bc360f08271', 7207980350),
    2021: ('06f242a5c6039486b47fb6272586b304dc6f7ec3d73c57aa0dd7a21aee144b81', 9116248703),
}

PAPER_FLOW = {'source_rows': 50674220, 'divorce_rows': 3321969, 'records': 3162013,
              'eligible_cases': 38333, 'candidate_claims': 34151, 'binary_claims': 3022,
              'binary_cases': 2601, 'candidate_records': 2610, 'candidate_cases': 2362,
              'retained_claims': 1826, 'retained_cases': 1696, 'review_queue': 774,
              'merged_duplicates': 10}

UNIT_COLUMNS = ['unit_id', 'cluster_id', 'weight', 'real_underdog', 'female_plaintiff',
                'post_divorce_action', 'record_length_chars', 'female_statement_chars',
                'male_statement_chars']


def _log(message):
    print(message, file=sys.stderr, flush=True)


# ============================================================================ stage 1: rows
CAUSE = re.compile('离婚|夫妻财产|婚内财产')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_archive(raw_dir, year):
    """The annual ZIP whose name starts with the year and whose size matches the release."""
    size = ARCHIVES[year][1]
    found = [p for p in Path(raw_dir).rglob(f'{year}*.zip') if p.stat().st_size == size]
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one {year} archive of {size} bytes under {raw_dir}')
    return found[0]


def scan_archive(path, year):
    """Yield every divorce-related CSV row of one annual ZIP, in member and row order."""
    csv.field_size_limit(64 * 1024 * 1024)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.lower().endswith('.csv'):
                continue
            with archive.open(member) as stream:
                sample = stream.read(32768)
            try:   # every member of the paper's archives is UTF-8 with BOM
                sample[:-4].decode('utf-8')
                encoding = 'utf-8-sig'
            except UnicodeDecodeError:
                encoding = 'gb18030'
            with archive.open(member) as binary, io.TextIOWrapper(binary, encoding=encoding, newline='') as text:
                reader = csv.DictReader(text)
                if not reader.fieldnames or '全文' not in reader.fieldnames:
                    raise ValueError(f'Unexpected archive schema in {member.filename}')
                for index, row in enumerate(reader):
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError(f'Malformed CSV row {member.filename}:{index}')
                    if CAUSE.search(row.get('案件名称', '') + ' ' + row.get('案由', '')):
                        yield {'archive_year': year, 'csv_member': member.filename,
                               'csv_row_index': index, 'raw': row}


def cache_divorce_rows(raw_dir, cache_dir, verify_hash=False):
    """Stage 1: write ``<cache_dir>/<year>.jsonl.gz`` for every year not yet cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        target = cache_dir / f'{year}.jsonl.gz'
        if target.exists():
            continue
        path = find_archive(raw_dir, year)
        if verify_hash and file_sha256(path) != ARCHIVES[year][0]:
            raise ValueError(f'{path.name}: SHA-256 differs from the archive used for the paper')
        partial = target.with_suffix('.gz.partial')
        kept = 0
        with gzip.open(partial, 'wt', encoding='utf8', compresslevel=5) as out:
            for record in scan_archive(path, year):
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                kept += 1
        partial.replace(target)
        _log(f'{year}: {kept} divorce-related rows cached')


def iter_divorce_rows(cache_dir):
    for year in YEARS:
        with gzip.open(Path(cache_dir) / f'{year}.jsonl.gz', 'rt', encoding='utf8') as f:
            for line in f:
                yield json.loads(line)


# ============================================================================ text helpers
def normalize(s):
    """NFKC, HTML-unescaped, tag- and whitespace-free text without the distributor watermark."""
    t = re.sub(r'\s+', '', unicodedata.normalize('NFKC', html.unescape(re.sub('<[^>]*>', '', s or ''))))
    return re.sub(r'(?:来自|更多数据[:：]?|百度搜索|关注公众号)?[“"「]?马克数据网[”"」]?(?:[:：]?www\.macrodatas\.cn)?$', '', t)


def digest(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def span(t, a, b):
    return dict(start=a, end=b, text=t[a:b])


def clauses(text, offset=0):
    return [dict(text=m.group(), start=offset + m.start(), end=offset + m.end())
            for m in re.finditer(r'[^。;；]+[。;；]?', text)]


# ---------------------------------------------------------------------------- money amounts
NUM = r'[0-9零〇○一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟萬億点.,]+'
MONEY = re.compile(r'(?<![0-9.])(' + NUM + r')(?:元|圆)')
DIGITS = dict(zip('零〇○一二三四五六七八九两壹贰叁肆伍陆柒捌玖', [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
UNITS = {'十': 10, '拾': 10, '百': 100, '佰': 100, '千': 1000, '仟': 1000, '万': 10000, '萬': 10000,
         '亿': 100000000, '億': 100000000}


def number(text):
    """Parse an Arabic or Chinese numeral; ambiguous mixed forms raise ValueError."""
    text = text.replace(',', '').replace('人民币', '')
    if re.fullmatch(r'\d+(?:\.\d+)?(?:万|亿)?', text):
        scale = 10000 if text.endswith('万') else 100000000 if text.endswith('亿') else 1
        return Decimal(text[:-1] if scale != 1 else text) * scale
    if re.search(r'[0-9.]', text):
        raise ValueError('Mixed numeral format')
    if '点' in text:
        a, b = text.split('点')
        scale = 1
        if b.endswith('万'):
            scale = 10000
            b = b[:-1]
        if not b or any(x not in DIGITS for x in b):
            raise ValueError('Bad decimal')
        return (number(a) + Decimal('0.' + ''.join(str(DIGITS[x]) for x in b))) * scale
    if not text:
        raise ValueError('Empty numeral')
    if len(text) >= 2 and text[-2] in '万萬千仟百佰' and text[-1] in DIGITS and DIGITS[text[-1]] != 0:
        raise ValueError('Omitted unit')
    total = section = n = 0
    if all(c in DIGITS for c in text):
        return Decimal(''.join(str(DIGITS[c]) for c in text))
    for c in text:
        if c in DIGITS:
            n = DIGITS[c]
        elif c in UNITS:
            u = UNITS[c]
            if u < 10000:
                section += (n or 1) * u
                n = 0
            elif u == 10000:
                section = (section + n) * u
                total += section
                section = n = 0
            else:
                total = (total + section + n) * u
                section = n = 0
        else:
            raise ValueError('Unsupported numeral')
    return Decimal(total + section + n)


def amounts(text):
    """All yuan amounts in fen, or None when any amount is ambiguous."""
    out = []
    for m in MONEY.finditer(text):
        if re.match(NUM + r'[角分]', text[m.end():]):
            return None
        try:
            v = number(m[1])
            fen = v * 100
        except (ValueError, InvalidOperation):
            return None
        if fen != fen.to_integral_value():
            return None
        out.append(dict(text=m.group(), amount_fen=int(fen), start=m.start(), end=m.end()))
    return out


# ============================================================================ stage 3: rules
DOWRY = (r'彩礼|彩禮|采礼|财礼|聘礼|聘金|聘财|嫁妆|嫁妝|嫁装|陪嫁|陪送|妆奁|嫁奁|嫁资|奁产|'
         r'(?:订婚|结婚|定亲|订亲)[^。;]{0,12}(?:礼金|礼钱|礼款|三金)')
ASSETS = {
    'housing': r'房屋|房产|住房|住宅|楼房|平房|北房|南房|东厢房|西厢房|房子|房款|房贷|购房|卖房|安置房|宅基地|拆迁|征收|按揭|还贷|公房',
    'savings': r'存款|存折|账户|帐户|银行卡|银行存|储蓄|公积金|理财|基金|股票|现金',
    'vehicle': r'车辆|汽车|轿车|机动车|摩托车|电动车|小客车',
    'equity': r'股权|股份|出资额|公司资产|公司财产|经营收益|经营收入|合伙',
    'debt': r'债务|债权|借款|欠款|贷款|借贷|共同债务',
    'land': r'土地|承包地|责任田|耕地|土地承包',
    'agriculture': r'农作物|树木|林木|果树|粮食|庄稼|牲畜|牛|羊|猪|藏獒|犬只',
    'chattels': r'家电|电器|家具|财物|物品|冰箱|洗衣机|空调|电视机|首饰|藏獒|建房原材料|建筑材料|铁皮|水泥|钉子',
    'settlement': r'补偿款|折价款|补偿金|补偿费|分割款|离婚协议|财产协议',
}
PROPERTY = '|'.join(ASSETS.values()) + r'|夫妻共同财产|共同财产|个人财产|婚前财产|财产分割'
ACTION = r'分割|分得|返还|交付|给付|支付|偿还|承担|撤销|履行|确认|过户|归.{0,16}所有|补偿'
OPPOSE = (r'不同意|不认可|不承认|否认|不属于|不属|不应|不可以|无权|驳回|不予支持|不能成立|不存在|没有依据|无.{0,5}依据|不属实|'
          r'所述.{0,12}(?:不实|不相符)|不知情|无效|显失公平|不合理|欺诈|胁迫|非.{0,12}真实意思|不是.{0,12}真实意思|'
          r'不是(?:夫妻)?共同财产|已(?:经)?(?:支付|给付|偿还|归还|分割|处理|履行)|(?:已|已经).{0,20}(?:花完|用于家庭(?:生活|开支))|'
          r'不愿(?:意)?(?:再)?(?:给付|支付|返还|分割|履行)|实际(?:数额|金额)(?:为|是)|婚前财产|个人财产|归我所有|应归被告|'
          r'数额过高|金额过高|只同意|只能|债权人.{0,20}(?:另行|另案)主张|(?:另行|另案)处理|与原告无关|'
          r'希望.{0,20}(?:维持|支持).{0,12}(?:原|原来).{0,8}协议|履行.{0,35}(?:后|才).{0,20}(?:支付|给付|协助)|'
          r'(?:只要|除非).{0,35}(?:就|才)|加入.{0,15}(?:名字|姓名)|加名|(?:是|系|属于)[^。;]{0,15}(?:父母|父亲|母亲|第三人)(?:的)?财产|'
          r'属于[^。;]{0,12}(?:父母|父亲|母亲)[^。;]{0,6}(?:所有|出资)|'
          r'要求[^。;]{0,30}(?:归我|归被告|给孩子|给子女|过户给|由原告承担|平分|平均分割|依法分割)|'
          r'应当[^。;]{0,25}(?:平分|平均分割|依法分割)|应由原告承担')
CHILD = r'抚养费|抚育费|教育费|探望|探视|抚养权|监护权|子女[^。;]{0,8}生活费'
NONPROPERTY_RELIEF = r'生活补助费?|经济帮助费?|精神损害(?:赔偿)?|损害赔偿|医疗费|医药费|护理费|治疗扶助金'
FACT_BOUNDARY = (r'经(?:本院)?(?:审理|庭审)(?:查明|认定)|本院(?:经审理)?(?:查明|认定如下|确认(?:以下|如下)(?:法律|案件)?事实)'
                 r'|经审理[,，:]?查明|经审查[,，:]?认定|经庭审举证|经庭审质证|经当庭(?:举证|质证)|本院组织.{0,15}质证|合议庭认为'
                 r'|本案(?:在)?审理过程中[,，:]?[^。;]{0,35}(?:无争议的事实|查明|确认)|(?:双方|原、被告)(?:均)?无争议的事实'
                 r'|(?:原告|被告)[^。;]{0,30}(?:为支持|为证明|提交|提供|出示)[^。;]{0,20}(?:证据|主张)|为证明[^。;]{0,25}证据|上述事实'
                 r'|根据(?:当事人|原、被告|双方)[^。;]{0,30}(?:陈述|举证)|本案(?:的)?争议焦点')
SUPPORT = r'(?<!不)予以支持|(?<!不)应予支持|依法支持|(?<!不)予支持|(?<!不)予以采纳'
DENY = r'不予支持|不予采纳|不予认可|不予分割|缺乏.{0,10}依据|无事实.{0,8}依据|不能成立|不予准许'
PAY = r'支付|给付|付给|偿还|返还|归还|补偿'
ANCILLARY_COST = r'诉讼费|受理费|保全费|公告费|鉴定费'
CHANGE = (r'原告[^。;]{0,110}(?:变更|增加|减少|撤回|放弃)[^。;]{0,100}(?:诉讼请求|诉请|请求|财产分割)'
          r'|原告[^。;]{0,110}(?:诉讼请求|诉请|请求|财产分割)[^。;]{0,100}(?:变更|增加|减少|撤回|放弃)'
          r'|原告[^。;]{0,110}(?:财产|请求|诉请)[^。;]{0,60}(?:另案处理|另行起诉|暂不分割|不再分割)'
          r'|原告[^。;]{0,110}(?:同意|要求)?[^。;]{0,25}(?:另案处理|另行起诉|暂不分割|不再分割)[^。;]{0,60}(?:财产|请求|诉请)'
          r'|原告[^。;]{0,45}(?:自愿承担|同意承担)[^。;]{0,30}(?:债务|欠款)')


def kinds(text):
    return sorted(k for k, p in ASSETS.items() if re.search(p, text))


def primary_kinds(text):
    """Asset categories of a request, resolving lexical overlaps to one asset claim."""
    ks = set(kinds(text))
    if {'agriculture', 'chattels'} <= ks and re.search(r'藏獒', text) and not re.search(
            r'农作物|树木|林木|果树|粮食|庄稼|牲畜|牛|羊|猪|犬只', text):
        ks.discard('agriculture')
    if 'chattels' in ks and re.search(r'(?:房屋|住房|住宅|室)内[^。;]{0,25}(?:生活用品|家用电器|家具|物品)', text):
        ks.discard('housing')
    # A compensation payment attached to another asset is the remedy for that asset.
    if 'settlement' in ks and len(ks) > 1 and re.search(PAY + r'|补偿|折价|分割款', text):
        ks.discard('settlement')
    if 'settlement' in ks and len(ks) > 1 and re.search(r'离婚协议|财产协议', text) and not re.search(r'补偿款|折价款|分割款', text):
        ks.discard('settlement')
    if 'savings' in ks and len(ks) > 1 and re.search(r'折成现金|折价(?:给付|补偿)|变现', text):
        ks.discard('savings')
    return sorted(ks)


class Exclude(Exception):
    pass


def need(x, reason):
    if not x:
        raise Exclude(reason)


def _headers(pre, limit):
    """Exactly one plaintiff and one defendant header, each with an explicit sex (男/女)."""
    block = pre[:min(limit, 2600)]
    out = {}
    for role in ['原告', '被告']:
        found = []
        for m in re.finditer(r'(?<!代理人)(?<!代理)' + role + r'[:：]?([^,，。:：;；()（）]{1,20})(?:[（(][^）)]{0,55}[）)])?[,，]([^。;；]{0,100}?)(男|女)(?=[,，。;；])', block):
            name = re.sub(r'^(?:系|为)', '', m[1])
            context = m.group()
            if re.search(r'委托代理|法定代理|诉讼代理', context[:max(0, context.find(name))]):
                continue
            found.append((name, m[3], m))
        unique = {(x[0], x[1]) for x in found}
        need(len(unique) == 1, 'not_one_explicit_sex_litigant_per_role')
        name, sex = next(iter(unique))
        m = next(x[2] for x in found if x[0] == name and x[1] == sex)
        out[role] = (name, sex, span(pre, m.start(), m.end()))
    need(out['原告'][0] != out['被告'][0] and out['原告'][1] != out['被告'][1], 'spouse_identity_or_sex_ambiguous')
    return out


def extract_case(source):
    """Screen one judgment; return the case record with status and exclusion reason."""
    raw = source['raw']
    t = normalize(raw.get('全文', ''))
    sha = digest(t)
    r = dict(case_id='cjo-claimcase-' + sha[:20], normalized_text_sha256=sha,
             case_number=normalize(raw.get('案号', '')), court=raw.get('法院', ''),
             judgment_date=raw.get('裁判日期', ''), status='excluded', reason=None)
    try:
        need(t, 'missing_full_text')
        need('民事' in raw.get('案件类型', '') or '民事判决书' in t[:350], 'not_civil')
        need('判决' in raw.get('文书类型', '') or '民事判决书' in t[:350], 'not_judgment')
        need(('一审' in raw.get('审理程序', '') or bool(re.search(r'民.{0,4}初', r['case_number'])))
             and not re.search(r'二审|再审', raw.get('审理程序', '')), 'not_first_instance')
        need(not re.search(DOWRY, t), 'dowry_or_bride_price')
        order_marks = list(re.finditer(r'判决如下[:：]?|判决[:：]', t))
        reason_marks = list(re.finditer(r'本院认为', t))
        need(order_marks, 'missing_current_order')
        om = order_marks[-1]
        rm = next((x for x in reversed(reason_marks) if x.start() < om.start()), None)
        need(rm, 'missing_current_reason')
        tail = t[om.end():]
        finish = re.search(r'案件受理费|本案受理费|诉讼费|如果未按|如未按|如不服|审判长|审判员|代理审判员|人民陪审员|附[:：]', tail)
        need(finish, 'missing_order_end')
        order_text = tail[:finish.start()]
        order_text = re.sub(r'(?<=[。;；])(?:本案|该案|本诉|(?:由)?(?:原告|被告)[^,，。;；]{0,18})$', '', order_text)
        need(order_text, 'empty_order')
        r['disposition'] = span(t, om.end(), om.end() + len(order_text))
        pre = t[:rm.start()]
        pm = list(re.finditer(r'原告[^。;；]{0,24}(?:诉称|起诉称|陈述称|称(?=[:,，：]))', pre))
        need(pm, 'missing_plaintiff_statement')
        pmark = pm[0]
        dm = next((x for x in re.finditer(r'被告[^。;；]{0,24}(?:辩称|答辩称|答辩认为|辩解称|答辩(?=[:,，：])|表示(?=不同意|同意|不认可|认可|要求|请求)|称(?=[:,，：]))', pre)
                   if x.start() > pmark.start()), None)
        need(dm, 'missing_defendant_statement')
        hs = _headers(pre, pmark.start())
        ps = pmark.start()
        ds = dm.start()
        transition = re.search(FACT_BOUNDARY, pre[ds:])
        de = ds + transition.start() if transition else len(pre)
        plaintiff = pre[ps:ds]
        defendant = pre[ds:de]
        audit = pre[de:]
        need(len(plaintiff) >= 20 and len(defendant) >= 12, 'statements_too_short')
        need(not re.search(r'未(?:作|做)?答辩|未提交答辩|未到庭|拒不到庭|缺席', defendant), 'no_substantive_defense')
        need(not re.search(r'第三人|反诉原告|反诉被告', t[:ps]), 'extra_party_or_counterclaim')
        need(not re.search(r'本院认为|本院认定|本院查明|经庭审质证', plaintiff + defendant), 'judicial_text_in_statements')
        r['sections'] = {'plaintiff': span(t, ps, ds), 'defendant': span(t, ds, de),
                         'facts_audit_only': span(t, de, rm.start()),
                         'reasoning_audit_only': span(t, rm.start(), om.start())}
        r.update(party_a_name=hs['原告'][0], party_a_gender=hs['原告'][1], party_b_name=hs['被告'][0],
                 party_b_gender=hs['被告'][1], female_party='A' if hs['原告'][1] == '女' else 'B',
                 header_evidence=[hs['原告'][2], hs['被告'][2]])
        head = raw.get('案由', '') + raw.get('案件名称', '') + t[:ps]
        post = bool(re.search(r'离婚后财产|离婚财产纠纷', head))
        grant = bool(re.search(r'(?<!不)(?:准予|准许|准).{0,45}离婚|解除.{0,45}婚姻关系', order_text))
        need(post or grant, 'not_postdivorce_or_granted_divorce')
        r['record_type'] = 'post_divorce_property' if post else 'same_judgment_divorce_property'
        need(re.search(PROPERTY, plaintiff), 'no_property_in_plaintiff_statement')
        r.update(post_statement_audit_text=audit, status='eligible_case', reason='claim_extraction_pending')
    except Exclude as e:
        r['reason'] = str(e)
    return r


def _request_units(text):
    """Candidate prayer spans: numbered items, narrative prayers and a trailing formal list."""
    units = []
    for intro in re.finditer(r'(?:诉讼请求|起诉请求|请求事项|请求法院(?:依法)?判令|请求判令|现请求|故请求|起诉要求|现诉请)[:：]?', text):
        end = len(text)
        stop = re.search(r'事实(?:和|与|及)?理由|事实理由|理由如下|为证明|原告(?:提交|提供)', text[intro.end():])
        if stop:
            end = intro.end() + stop.start()
        block = text[intro.end():end]
        marks = list(re.finditer(r'(?:^|(?<=[,，:：;；]))[一二三四五六七八九十0-9]+[、.]', block))
        if marks:
            for i, m in enumerate(marks):
                a = m.end()
                b = marks[i + 1].start() if i + 1 < len(marks) else len(block)
                units.append((intro.end() + a, intro.end() + b, block[a:b]))
        else:
            for c in clauses(block, intro.end()):
                units.append((c['start'], c['end'], c['text']))
    for c in clauses(text):
        ms = list(re.finditer(r'请求|要求|诉请|判令|依法分割|诉至法院', c['text']))
        if not ms:
            continue
        m = ms[-1]
        units.append((c['start'] + m.start(), c['end'], c['text'][m.start():]))
    formal = list(re.finditer(r'(?:现|故|特|遂)?(?:诉至法院|提起诉讼|起诉)?[,，]?(?:请求|要求|诉请)|诉讼请求|起诉请求', text))
    if formal:
        base = formal[-1].start()
        tail = text[base:]
        boundaries = {0, len(tail)}
        for m in re.finditer(r'[。;；]|(?<=[,，:：])(?=[一二三四五六七八九十0-9]+[、.])', tail):
            boundaries.add(m.end())
        bs = sorted(boundaries)
        for a, b in zip(bs, bs[1:]):
            raw_piece = tail[a:b]
            piece = re.sub(r'^[,，:：;；一二三四五六七八九十0-9、.]+', '', raw_piece)
            shift = raw_piece.find(piece)
            if re.search(PROPERTY, piece) and re.search(ACTION, piece):
                units.append((base + a + shift, base + a + shift + len(piece), piece))
    out = []
    seen = set()
    for a, b, s in units:
        original = s
        s = re.sub(r'^[,，:：;；]+|[,，:：;；]+$', '', s)
        a += original.find(s)
        b = a + len(s)
        if not s:
            continue
        key = re.sub(r'[，,。;；:：]', '', s)
        if key in seen:
            continue
        if any(max(0, min(b, y[1]) - max(a, y[0])) / max(1, min(b - a, y[1] - y[0])) >= .7 for y in out):
            continue
        seen.add(key)
        out.append((a, b, s))
    return out


def _relief_amount(text):
    vals = [v for v in (amounts(text) or []) if not re.search(
        ANCILLARY_COST, text[max(0, v['start'] - 18):min(len(text), v['end'] + 5)])]
    if not vals:
        return None, vals
    candidates = []
    for v in vals:
        window = text[max(0, v['start'] - 28):min(len(text), v['end'] + 12)]
        if re.search(PAY + r'|补偿款|折价款|分割款|差价', window):
            candidates.append(v)
    if len(candidates) == 1:
        return candidates[0]['amount_fen'], vals
    if len(vals) == 1:
        v = vals[0]
        window = text[max(0, v['start'] - 28):min(len(text), v['end'] + 12)]
        broad_window = text[max(0, v['start'] - 90):min(len(text), v['end'] + 20)]
        valuation = bool(re.search(r'价值|估价|暂估|市场价|市价|总价|原价|购置价|作价|约合', window))
        if not valuation and (re.search(PAY + r'|补偿|折价|退赔', broad_window)
                              or re.search(r'(?:折合|折价)[^。;]{0,18}(?:给|归)(?:原告|我)', broad_window)):
            return v['amount_fen'], vals
        if re.search(r'存款|现金|债权|债务|借款|欠款|贷款|公积金|工资|收益|收入', window) and not re.search(r'价值|估价|总价|房价', window):
            return v['amount_fen'], vals
    return None, vals


def _directed_payment(text, payer, payee, before=80, after=45):
    """Detect payer -> payment verb -> payee without crossing the payee first."""
    for payment in re.finditer(PAY, text):
        left = text[max(0, payment.start() - before):payment.start()]
        payer_matches = list(re.finditer(payer, left))
        if not payer_matches:
            continue
        payer_match = payer_matches[-1]
        if re.search(payee, left[payer_match.end():]):
            continue
        right = text[payment.end():payment.end() + after]
        if re.search(payee, right):
            return True
    return False


def _located_property_targets(text):
    marks = list(re.finditer(r'位于|坐落于', text))
    if len(marks) < 2:
        return []
    candidates = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        raw = text[m.start():end]
        trimmed = re.sub(r'(?:以及|并且|和|及|、|,|，)+$', '', raw)
        if not re.search(ASSETS['housing'] + r'|' + ASSETS['land'], trimmed):
            continue
        candidates.append((m.start(), m.start() + len(trimmed), trimmed))
    return candidates if len(candidates) >= 2 else []


def _enumerated_chattel_targets(text):
    item = r'冰箱|洗衣机|空调|电视机|电视|沙发|茶几|衣柜|床|餐桌|书桌|电脑|电动车|摩托车|缝纫机|电饭锅|电磁炉|热水器|饮水机|梳妆台|音响|柜子'
    counter = r'[一二三四五六七八九十百0-9]+(?:台|个|张|套|床|件|辆|架|组|只|把|顶|支)'
    found = [(m.start(), m.end(), m.group()) for m in re.finditer(r'(?:' + item + r')[^、,，;；。]{0,12}?' + counter, text)]
    return found if len(found) >= 2 else []


def _heterogeneous_asset_targets(text, asset_kinds):
    if len(asset_kinds) < 2:
        return []
    out = []
    for kind in asset_kinds:
        matches = list(re.finditer(ASSETS[kind], text))
        if len(matches) != 1:
            return []
        m = matches[0]
        left = max(text.rfind('、', 0, m.start()), text.rfind(';', 0, m.start()), text.rfind('；', 0, m.start())) + 1
        rights = [x for x in (text.find('、', m.end()), text.find(';', m.end()), text.find('；', m.end())) if x >= 0]
        right = min(rights) if rights else len(text)
        component = text[left:right].strip(' ,，:：。')
        shift = text[left:right].find(component)
        ca = left + max(shift, 0)
        cb = ca + len(component)
        if set(primary_kinds(component)) != {kind}:
            ca, cb, component = m.start(), m.end(), m.group()
        out.append((ca, cb, component, kind))
    return out


def _claim_link_text(claim):
    return claim.get('target_text') or claim['text']


def _specific_open_form_target(claim):
    text = _claim_link_text(claim)
    if claim.get('atomic_subclaim'):
        return True
    ks = set(claim.get('asset_kinds') or [])
    if not ks:
        return False
    if ks == {'debt'} and re.search(r'共同财产', text) and claim.get('requested_amount_fen') is None:
        return False
    return True


def extract_claims(case):
    """Current plaintiff property prayers; bundled prayers also get automatic sub-claims."""
    if case['status'] != 'eligible_case':
        return []
    ps = case['sections']['plaintiff']
    out = []
    request_text = ps['text']
    evidence = re.search(r'原告[^。;]{0,30}(?:为支持|为证明|提交|提供|出示)[^。;]{0,20}(?:证据|主张)', request_text)
    if evidence:
        request_text = request_text[:evidence.start()]
    for a, b, text in _request_units(request_text):
        if not re.search(PROPERTY, text) or not re.search(ACTION, text):
            continue
        if re.search(r'无共同财产|没有共同财产|不存在共同财产', text):
            continue
        if re.search(CHILD, text) and not re.search(PROPERTY, text):
            continue
        ks = primary_kinds(text)
        amount, vals = _relief_amount(text)
        remedies = []
        for name, pat in [('payment', PAY + r'|补偿'), ('ownership', r'归.{0,16}所有|分割'), ('return', r'返还|归还|交付'),
                          ('transfer', r'过户'), ('revoke', r'撤销'), ('confirm', r'确认'),
                          ('debt_allocation', r'承担.{0,20}债务|债务.{0,20}承担')]:
            if re.search(pat, text):
                remedies.append(name)
        start = ps['start'] + a
        end = ps['start'] + b
        cid = 'cjo-claim-' + digest(case['case_id'] + '|' + str(start) + '|' + text)[:20]
        title_to_a = bool(re.search(r'归(?:原告|我).{0,10}所有|返还(?:给)?(?:原告|我)|过户.{0,25}(?:原告|我)|登记.{0,20}(?:原告|我).{0,8}名下', text))
        payment_direction = (
            'A_TO_B' if _directed_payment(text, r'(?:原告|我)', r'(?:被告|对方)', 22, 22)
            else 'B_TO_A' if _directed_payment(text, r'(?:被告|对方)', r'(?:原告|我)', 30, 30) or re.search(r'判令被告[^。;]{0,30}(?:支付|给付|补偿|返还)', text)
            else None)
        beneficiary = ('A' if title_to_a else 'B' if payment_direction == 'A_TO_B' or re.search(r'归被告.{0,8}所有|由被告所有', text)
                       else 'A')
        mixed_nonproperty = bool(re.search(NONPROPERTY_RELIEF, text))
        located_assets = re.findall(r'(?:位于|坐落于)[^,，;；。]{2,80}(?:房屋|房产|住房|住宅)', text)
        financial_subtypes = set(re.findall(r'工资(?:收入)?|公积金|养老保险(?:金|费)?|股票|基金|存款|理财(?:产品)?|现金', text))
        repeated_realty = bool(re.search(
            r'(?:房屋|房产|住房|住宅|宅基地|土地)[^。;]{0,90}(?:和|及|以及|、)[^。;]{0,90}(?:房屋|房产|住房|住宅|宅基地|土地)', text))
        multi_asset = bool(
            len(ks) > 1 or len(financial_subtypes) >= 2
            or re.search(r'两套|两辆|分别|甲地|乙地|第一套|第二套', text)
            or re.search(r'(?:汽车|车辆|轿车)[^。;]{0,30}保险费|保险费[^。;]{0,30}(?:汽车|车辆|轿车)', text)
            or re.search(r'安置费[^。;]{0,35}(?:房屋|住房|住宅)|(?:房屋|住房|住宅)[^。;]{0,35}安置费', text)
            or len(located_assets) >= 2 or repeated_realty)
        enumerated_items = re.findall(r'[一二三四五六七八九十百0-9]+(?:台|个|张|套|床|件|支|顶|辆|架|组|只|把|间|座|幢|处)', text)
        multi_chattel = bool(len(enumerated_items) >= 2)
        broad_bundle = bool(re.search(r'全部|所有(?:的)?(?:共同)?财产|(?:其他|其它|其余)(?:共同)?财产|生活用品|家用电器|家具家电|电器类|家具类', text))
        remedy_groups = set()
        if set(remedies) & {'ownership', 'transfer', 'confirm', 'return'}:
            remedy_groups.add('title_or_delivery')
        if 'revoke' in remedies:
            remedy_groups.add('revoke')
        if 'debt_allocation' in remedies:
            remedy_groups.add('debt_allocation')
        parent = dict(claim_id=cid, case_id=case['case_id'], claim_index=len(out) + 1, text=text, start=start, end=end,
                      asset_kinds=ks, remedies=remedies, requested_amount_fen=amount, amounts=vals,
                      requested_beneficiary=beneficiary, requested_payment_direction=payment_direction,
                      mixed_nonproperty=mixed_nonproperty,
                      compound_claim=len(remedy_groups) > 1 or multi_asset or multi_chattel or broad_bundle or bool(vals and len(vals) > 1)
                      or bool(vals and re.search(r'\+|另加|外加', text)) or bool(re.search(r'利息|违约金', text)),
                      atomic_subclaim=False, has_atomic_children=False)
        out.append(parent)
        targets = _located_property_targets(text)
        if not targets and ks == ['chattels']:
            targets = _enumerated_chattel_targets(text)
        target_kinds = None
        if not targets:
            heterogeneous = _heterogeneous_asset_targets(text, ks)
            if heterogeneous:
                targets = [x[:3] for x in heterogeneous]
                target_kinds = [x[3] for x in heterogeneous]
        if targets:
            parent['has_atomic_children'] = True
            parent['compound_claim'] = True
            for target_index, (ta, tb, target) in enumerate(targets):
                child_kinds = [target_kinds[target_index]] if target_kinds else primary_kinds(target)
                if not child_kinds:
                    continue
                child_amount, child_vals = _relief_amount(target)
                child_beneficiary = ('A' if re.search(r'归(?:原告|我).{0,12}所有|返还(?:给)?(?:原告|我)', target)
                                     else 'B' if re.search(r'归被告.{0,12}所有|由被告所有', target) else beneficiary)
                child = dict(parent)
                child.update(
                    claim_id='cjo-claim-' + digest(case['case_id'] + '|atomic|' + str(start + ta) + '|' + target)[:20],
                    parent_claim_id=parent['claim_id'], atomic_subclaim=True, has_atomic_children=False,
                    target_text=target, target_start=start + ta, target_end=start + tb,
                    atomic_split_method='asset_kind' if target_kinds else 'explicit_item_or_location',
                    asset_kinds=child_kinds, requested_amount_fen=child_amount, amounts=child_vals,
                    requested_beneficiary=child_beneficiary, mixed_nonproperty=False, compound_claim=False)
                out.append(child)
    for i, claim in enumerate(out, 1):
        claim['claim_index'] = i
    return out


def _shared_signal(a, b):
    score = 0
    ka = set(kinds(a))
    kb = set(kinds(b))
    score += 2 * len(ka & kb)
    aa = amounts(a) or []
    bb = amounts(b) or []
    if {x['amount_fen'] for x in aa} & {x['amount_fen'] for x in bb}:
        score += 4

    def tokens(s):
        return set(re.findall(r'[A-Za-z0-9×*]{3,}|[一-龥]{4,}', re.sub(r'原告|被告|请求|要求|依法|共同财产|个人财产|诉讼请求|本院|认为', '', s)))
    if tokens(a) & tokens(b):
        score += 4

    def compact(s):
        return re.sub(r'原告|被告|请求|要求|依法|共同|财产|所有|支付|给付|返还|分割|补偿|本院|认为|支持|诉讼', '', s)
    ca, cb = compact(a), compact(b)
    if min(len(ca), len(cb)) >= 8 and SequenceMatcher(None, ca, cb).ratio() >= .34:
        score += 2
    return score


def _exact_amounts(text):
    return {x['amount_fen'] for x in (amounts(text) or [])}


def _measure_values(text):
    out = {}
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(平方米|平米|亩|%|%|间|套|辆|座|幢)', text):
        unit = {'平米': '平方米', '%': '%'}.get(m[2], m[2])
        out.setdefault(unit, set()).add(float(m[1]))
    for m in re.finditer(r'百分之(\d+(?:\.\d+)?)', text):
        out.setdefault('%', set()).add(float(m[1]))
    if re.search(r'二分之一|一半', text):
        out.setdefault('%', set()).add(50.0)
    return out


def _distinctive_tokens(text, n=5):
    cleaned = re.sub(r'原告|被告|原、被告|双方|婚姻关系存续期间|夫妻共同财产|共同财产|个人财产|婚前财产|夫妻共同债务|共同债务|共同债权|诉讼请求|依法|要求|请求|判令|分割|支付|给付|返还|归还|所有|约定|本院|认为|予以支持|不予支持|房屋|房产|住房|住宅|存款|车辆|汽车|轿车|债务|债权|平均|一半|各自|份额', '', text)
    out = set(re.findall(r'[A-Za-z0-9×*\-]{4,}', cleaned))
    for seq in re.findall(r'[一-龥]{' + str(n) + r',}', cleaned):
        out.update(seq[i:i + n] for i in range(len(seq) - n + 1))
    return out


def _strong_link(claim, text, allow_unique_category=False):
    requested = {x['amount_fen'] for x in (claim.get('amounts') or [])}
    if requested and requested & _exact_amounts(text):
        return True
    claim_text = _claim_link_text(claim)
    if claim.get('target_text'):
        core = re.sub(r'(?:均|分别)?归(?:原告|被告|我)[^。;]{0,16}所有|依法分割|平均分割|共同分割|[，,。;；]', '', claim_text)
        other = re.sub(r'[，,。;；]', '', text)
        minimum = 2 if claim.get('atomic_split_method') == 'asset_kind' else 4
        if len(core) >= minimum and core in other:
            return True
        return False
    if _distinctive_tokens(claim_text) & _distinctive_tokens(text):
        return True
    if allow_unique_category and _distinctive_tokens(claim_text, 3) & _distinctive_tokens(text, 3):
        return True
    shared = set(claim['asset_kinds']) & set(kinds(text))
    return bool(allow_unique_category and len(claim['asset_kinds']) == 1 and shared)


def link_opposition(case, claim, all_claims):
    """Best defendant clause opposing this property claim, or None."""
    ds = case['sections']['defendant']
    best = None
    samekind = sum(bool(set(x['asset_kinds']) & set(claim['asset_kinds'])) for x in all_claims if not x.get('has_atomic_children'))
    for s in clauses(ds['text'], ds['start']):
        # Remove only the divorce-position phrase before looking for property opposition.
        property_position = re.sub(
            r'(?:不(?:同意|愿意|愿|想)(?:与[^。;]{0,12})?离婚|不同意原告[^。;]{0,8}离婚(?:的)?(?:请求|诉请)?|(?<!不)同意(?:与[^。;]{0,12})?离婚)',
            '', s['text'])
        asset_word = r'(?:房屋|房产|住房|住宅|车辆|汽车|轿车|存款|公积金|股权|股份|土地|承包地|补偿款|折价款|财产)'
        agreement = re.search(
            r'(?<!不)同意(?:将)?[^。;]{0,45}' + asset_word + r'[^。;]{0,28}(?:归|给|交付|过户|分割)'
            r'|' + asset_word + r'[^。;]{0,28}(?<!不)同意(?:归|给|交付|过户|分割)',
            property_position)
        if agreement:
            agreed_text = property_position[agreement.start():agreement.end() + 20]
            trailing = property_position[agreement.end():]
            different_beneficiary = bool(re.search(r'原告[^。;]{0,10}(?:和|及|与)(?:子女|女儿|儿子|孩子)', agreed_text))
            same_relief_objection = bool(re.search(
                r'(?:但|但是|不过|而)[^。;]{0,15}不(?:同意|认可|应当)[^。;]{0,35}(?:' + asset_word + r'|归|给|交付|过户|分割|返还|补偿|折价|支付|给付)',
                trailing))
            conditional_money = bool(re.search(r'(?:但|但是|不过|而)[^。;]{0,35}(?:补偿|折价|支付|给付|金额|价款)', trailing))
            if not different_beneficiary and not same_relief_objection and not conditional_money:
                continue
        if re.search(r'(?:现在|现)[^。;]{0,24}(?:仍然|仍)?(?<!不)同意将[^。;]{0,45}(?:归|给|交付|过户给)原告(?:所有)?', property_position) and not re.search(
                r'(?:归|给|交付|过户给)原告[^。;]{0,12}(?:和|及|与)(?:子女|女儿|儿子|孩子)', property_position):
            continue
        if not re.search(OPPOSE, property_position):
            continue
        claim_text = _claim_link_text(claim)
        score = _shared_signal(claim_text, property_position)
        if _distinctive_tokens(claim_text, 4) & _distinctive_tokens(property_position, 4):
            score += 4
        if claim.get('requested_amount_fen') is not None and re.search(r'实际(?:数额|金额)(?:为|是)|(?:已|已经).{0,20}(?:花完|用于家庭(?:生活|开支))|数额过高|金额过高', s['text']):
            score = max(score, 2)
        blanket = bool(re.search(r'(?:(?:不同意|驳回|不认可|不予支持)[^。;]{0,25}(?:原告[^。;]{0,8})?(?:诉讼请求|诉请|请求)|(?:其他|其余|全部|所有)?(?:诉讼请求|诉请|请求)[^。;]{0,18}(?:不能成立|不予支持))', property_position))
        blanket_all = bool(re.search(r'(?:(?:不同意|驳回|不认可|不予支持)[^。;]{0,35}(?:全部|所有)(?:诉讼请求|诉请|请求)|(?:全部|所有)(?:诉讼请求|诉请|请求)[^。;]{0,18}(?:不能成立|不予支持))', property_position))
        collective = bool(claim.get('atomic_subclaim') and set(claim['asset_kinds']) & set(kinds(property_position)) and (
            re.search(r'上述|以上|全部|所有|均|这些|该批', property_position)
            or ('chattels' in claim['asset_kinds'] and re.search(r'家具|家电|电器|物品|财物|共同财产', property_position))))
        if blanket and case['record_type'] == 'post_divorce_property':
            score = max(score, 2)
        if blanket and re.search(PROPERTY, s['text']):
            score = max(score, 2)
        if blanket_all:
            score = max(score, 2)
        if collective:
            score = max(score, 2)
        if len(all_claims) == 1 and re.search(PROPERTY + r'|财产', s['text']):
            score = max(score, 2)
        if samekind > 1 and score < 6 and not blanket:
            continue
        if score < 2 and not (samekind == 1 and set(claim['asset_kinds']) & set(kinds(s['text']))):
            continue
        if best is None or score > best[0]:
            best = (score, s, blanket)
    if best:
        return dict(score=best[0], evidence=best[1], blanket=best[2])
    return None


def _order_parts(case):
    return clauses(case['disposition']['text'], case['disposition']['start'])


def _plaintiff_result_clause(claim, s, result_pattern, allow_unique_category=False):
    """True only when a result is expressly about this plaintiff claim."""
    text = s['text']
    if not re.search(PROPERTY, text):
        return False
    requested = {x['amount_fen'] for x in (claim.get('amounts') or [])}
    mentioned = _exact_amounts(text)
    if requested and mentioned and not (requested & mentioned):
        return False
    if not _strong_link(claim, text, allow_unique_category=allow_unique_category):
        return False
    for m in re.finditer(result_pattern, text):
        before = text[:m.start()]
        a_positions = [x.start() for x in re.finditer(
            r'(?:对|关于)原告[^。;]{0,35}(?:请求|主张|诉请|诉求|要求|诉称)'
            r'|原告[^。;]{0,28}(?:请求|主张|诉请|诉求|要求|诉称)'
            r'|原告的?(?:该项|上述)?(?:诉讼)?(?:请求|主张|诉请|诉求)'
            r'|(?:该项|上述)(?:诉讼)?(?:请求|主张|诉请|诉求)', before)]
        if not a_positions:
            continue
        b_positions = [x.start() for x in re.finditer(r'被告[^。;]{0,35}(?:抗辩|辩称|辩解|意见|主张|请求|诉请|要求)', before)]
        if b_positions and max(b_positions) > max(a_positions):
            continue
        return True
    return False


def label_claim(case, claim, all_claims):
    """Claim-level court outcome: FULLY_UPHELD, DENIED, a partial exclusion or unresolved."""
    result = dict(claim_id=claim['claim_id'], case_id=case['case_id'], status='unresolved', reason='outcome_not_linked',
                  court_outcome=None, GT_party=None, court_female_support=None, opposition=None, court_evidence=None,
                  reasoning_evidence=None)
    if claim.get('has_atomic_children'):
        result['reason'] = 'superseded_by_atomic_subclaims'
        return result
    if re.search(CHANGE, claim['text']):
        result['reason'] = 'claim_changed_or_withdrawn'
        return result
    opp = link_opposition(case, claim, all_claims)
    result['opposition'] = opp
    if not opp:
        result['reason'] = 'no_claim_level_property_opposition'
        return result
    if claim.get('mixed_nonproperty'):
        result['reason'] = 'property_not_isolated_from_divorce_or_child_claim'
        return result
    audit_sections = [*clauses(case['post_statement_audit_text']), *clauses(case['sections']['reasoning_audit_only']['text'])]
    change_evidence = next((s for s in audit_sections if re.search(CHANGE, s['text']) and (
        _strong_link(claim, s['text'], allow_unique_category=len(all_claims) == 1)
        or (len(all_claims) == 1 and re.search(PROPERTY, s['text'])))), None)
    if change_evidence:
        result['reason'] = 'claim_changed_or_withdrawn'
        return result
    order = case['disposition']['text']
    parts = _order_parts(case)
    reason = case['sections']['reasoning_audit_only']
    # In a post-divorce property suit, a merits dismissal of all current requests
    # disposes of each current property claim independently.
    substantive_parts = [p for p in parts if re.search(PAY + r'|归.{0,25}所有|返还|交付|过户|撤销|确认', p['text']) and not re.search(CHILD + r'|诉讼费|受理费', p['text'])]
    all_dismissal = next((p for p in parts if re.search(r'驳回[^。;；]{0,35}(?:(?:全部|所有)(?:诉讼请求|诉请)|(?:原告[^。;；]{0,16})?(?:诉讼请求|诉请))', p['text']) and not re.search(r'其他|其它|其余', p['text'])), None)
    merits_all = case['record_type'] == 'post_divorce_property' and bool(all_dismissal) and not substantive_parts
    if merits_all:
        result.update(status='labeled', reason='all_current_requests_merits_dismissal', court_outcome='DENIED', GT_party='B', court_evidence=all_dismissal)
    else:
        linked_reason = []
        unique_kind = bool(claim['asset_kinds']) and sum(
            bool(set(x['asset_kinds']) & set(claim['asset_kinds'])) for x in all_claims if not x.get('has_atomic_children')) == 1
        for s in clauses(reason['text'], reason['start']):
            if not re.search(SUPPORT + '|' + DENY, s['text']):
                continue
            unique_category = unique_kind and re.search(r'(?:原告|原、被告|双方)(?:(?!被告).){0,35}(?:请求|主张|诉求|要求)', s['text'])
            if _strong_link(claim, s['text'], allow_unique_category=bool(unique_category)):
                linked_reason.append(s)
        deny_reason = next((s for s in linked_reason if _plaintiff_result_clause(claim, s, DENY, unique_kind)), None)
        support_reason = next((s for s in linked_reason if _plaintiff_result_clause(claim, s, SUPPORT, unique_kind)), None)
        if len(all_claims) == 1 and _strong_link(claim, reason['text'], allow_unique_category=True):
            if not support_reason and _plaintiff_result_clause(claim, reason, SUPPORT, True):
                support_reason = reason
            if not deny_reason and _plaintiff_result_clause(claim, reason, DENY, True):
                deny_reason = reason
        residual = next((p for p in parts if re.search(r'驳回[^。;]{0,35}(?:其他|其它|其余)(?:诉讼请求|诉请|请求)', p['text'])), None)
        specific = next((p for p in parts if re.search(r'驳回', p['text']) and (_strong_link(claim, p['text'], allow_unique_category=len(all_claims) == 1) or (len(all_claims) == 1 and re.search(PROPERTY, p['text'])))), None)
        linked_substantive = [p for p in substantive_parts if _strong_link(claim, p['text'], allow_unique_category=len(all_claims) == 1)]
        b_party_pattern = r'(?:被告|' + re.escape(case['party_b_name']) + r')'
        a_party_pattern = r'(?:原告|' + re.escape(case['party_a_name']) + r')'
        directed_pay_parts = [p for p in substantive_parts if _directed_payment(p['text'], b_party_pattern, a_party_pattern)]
        possible_partial = bool(claim.get('requested_amount_fen') is not None and any(
            0 < v['amount_fen'] < claim['requested_amount_fen'] for p in directed_pay_parts for v in (amounts(p['text']) or [])))
        sole_claim_has_property_award = len(all_claims) == 1 and bool(substantive_parts)
        clean_full_denial = bool(deny_reason and not re.search(r'部分|超过[^。;]{0,16}部分|其余|过高', deny_reason['text']))
        if not claim['compound_claim'] and clean_full_denial and (residual or specific) and not linked_substantive and not possible_partial and not sole_claim_has_property_award:
            result.update(status='labeled', reason='linked_reasoning_denial_and_dismissal_order', court_outcome='DENIED', GT_party='B',
                          court_evidence=specific or residual, reasoning_evidence=deny_reason)
        elif not claim['compound_claim'] and case['record_type'] == 'same_judgment_divorce_property' and residual and not substantive_parts:
            result.update(status='labeled', reason='property_claim_with_residual_merits_dismissal', court_outcome='DENIED', GT_party='B', court_evidence=residual)
        elif not claim['compound_claim'] and clean_full_denial and not linked_substantive and not possible_partial and not sole_claim_has_property_award:
            result.update(status='labeled', reason='explicit_claim_denial_in_court_reasoning', court_outcome='DENIED', GT_party='B',
                          court_evidence=deny_reason, reasoning_evidence=deny_reason)
        else:
            matches = []
            measure_mismatches = []
            requested_measures = _measure_values(_claim_link_text(claim))
            for p in parts:
                if re.search(CHILD + r'|' + NONPROPERTY_RELIEF + r'|诉讼费|受理费|迟延履行|驳回|不予支持|不予处理', p['text']):
                    continue
                score = _shared_signal(claim['text'], p['text'])
                if claim['requested_amount_fen'] is not None:
                    vals = amounts(p['text']) or []
                    for v in vals:
                        if not re.search(PAY, p['text']):
                            continue
                        if not re.search(r'(?:被告|' + re.escape(case['party_b_name']) + r').{0,80}(?:' + PAY + r').{0,45}(?:原告|' + re.escape(case['party_a_name']) + r')', p['text']):
                            continue
                        matches.append((v['amount_fen'], p, score, 'money'))
                a_party = r'(?:原告|' + re.escape(case['party_a_name']) + r')'
                object_to_a = re.search(r'([^。;；]{0,180}?)(?:归' + a_party + r'.{0,8}所有|返还(?:给)?' + a_party + r'|过户.{0,35}' + a_party + r'|协助' + a_party + r'.{0,180}(?:过户|变更登记)|变更登记.{0,45}' + a_party + r'.{0,10}名下|(?:由|确认)?' + a_party + r'.{0,20}(?:享有|取得).{0,30}(?:产权|所有权|份额)|(?:债权|存款|款项).{0,40}(?:由)?' + a_party + r'.{0,12}享有|(?:由)?' + a_party + r'.{0,12}享有.{0,25}(?:债权|存款|款项|元)|' + a_party + r'.{0,12}名下)', p['text'])
                if object_to_a and _strong_link(claim, object_to_a.group(), allow_unique_category=sum(bool(set(x['asset_kinds']) & set(claim['asset_kinds'])) for x in all_claims if not x.get('has_atomic_children')) == 1):
                    awarded_measures = _measure_values(p['text'])
                    conflicts = {u for u in requested_measures.keys() & awarded_measures.keys() if not requested_measures[u] & awarded_measures[u]}
                    if conflicts:
                        measure_mismatches.append((p, sorted(conflicts)))
                    else:
                        matches.append((None, p, score, 'in_kind'))
            if claim['requested_amount_fen'] is not None:
                exact = [m for m in matches if m[0] == claim['requested_amount_fen']]
                partial = [m for m in matches if m[0] is not None and 0 < m[0] < claim['requested_amount_fen']]
                topic_link = bool(set(claim['asset_kinds']) & set(primary_kinds(exact[0][1]['text']))) if exact else False
                if len(exact) == 1 and claim.get('requested_beneficiary') == 'A' and not claim['compound_claim'] and (topic_link or support_reason or len(all_claims) == 1):
                    result.update(status='labeled', reason='exact_claim_amount_and_linked_order', court_outcome='FULLY_UPHELD', GT_party='A',
                                  court_evidence=exact[0][1], reasoning_evidence=support_reason)
                elif partial:
                    result.update(status='excluded_partial', reason='partial_amount_user_excluded', court_outcome='PARTLY_UPHELD',
                                  court_evidence=partial[0][1])
            if result['status'] == 'unresolved' and measure_mismatches:
                result.update(status='excluded_partial', reason='partial_quantity_or_share_user_excluded', court_outcome='PARTLY_UPHELD',
                              court_evidence=measure_mismatches[0][0])
            if result['status'] == 'unresolved':
                ink = [m for m in matches if m[3] == 'in_kind']
                unique_kind = bool(claim['asset_kinds']) and sum(bool(set(x['asset_kinds']) & set(claim['asset_kinds'])) for x in all_claims if not x.get('has_atomic_children')) == 1
                reciprocal = any(_directed_payment(p['text'], a_party_pattern, b_party_pattern, 70, 35) for p in parts)
                exclusive_to_a = bool(re.search(r'归(?:原告|我).{0,12}(?:一人|个人)?所有|(?:原告|我).{0,12}(?:个人|单独)所有|返还(?:给)?(?:原告|我)', claim['text']))
                exclusive_to_b = bool(re.search(r'归被告.{0,12}(?:一人|个人)?所有|由被告所有', claim['text']))
                open_form = bool(
                    re.search(r'分割|共同承担|各自承担|依法承担|共同偿还', claim['text'])
                    and not re.search(r'平均|一半|二分之一|百分之|\d+(?:\.\d+)?%|[一二三四五六七八九十]成', claim['text'])
                    and not exclusive_to_a and not exclusive_to_b
                    and claim.get('requested_amount_fen') is None and not claim['compound_claim']
                    and _specific_open_form_target(claim))
                equal_form = bool(
                    re.search(r'平均分割|各半|一人一半|各(?:占|享有|承担)\s*(?:50%|二分之一)|分割一半|二分之一', claim['text'])
                    and claim.get('requested_amount_fen') is None and not claim['compound_claim']
                    and _specific_open_form_target(claim))
                directly_disposed = next((p for p in linked_substantive if re.search(
                    r'归[^。;]{0,25}所有|(?:原告|被告|双方|原、被告)[^。;]{0,35}(?:享有|分得|承担|偿还)|共同共有|按份共有|各自负责', p['text'])), None)
                a_party = r'(?:原告|' + re.escape(case['party_a_name']) + r')'
                b_party = r'(?:被告|' + re.escape(case['party_b_name']) + r')'
                a_title_or_share = any(re.search(
                    r'归' + a_party + r'.{0,10}所有|' + a_party + r'.{0,25}(?:享有|分得)|' + a_party + r'.{0,20}(?:与|、)' + b_party + r'.{0,35}(?:各|共同)', p['text'])
                    for p in linked_substantive)
                b_pays_a = any(_directed_payment(p['text'], b_party, a_party) for p in substantive_parts)
                debt_shared = ('debt' in claim['asset_kinds'] and directly_disposed and bool(re.search(
                    b_party + r'.{0,30}(?:承担|偿还)|原、被告.{0,25}(?:各|共同).{0,15}(?:承担|偿还)', directly_disposed['text'])))
                benefits_a = bool(a_title_or_share or b_pays_a or debt_shared)
                equal_in_order = bool(re.search(r'各(?:占|享有|承担)[^。;]{0,12}(?:50%|二分之一|一半)|平均分割|各半|折价款[^。;]{0,12}(?:一半|二分之一|50%)', order))
                value_amounts = {x['amount_fen'] for x in (claim.get('amounts') or [])}
                half_value_paid = any(
                    any(abs(v['amount_fen'] * 2 - total) <= 1 for total in value_amounts)
                    for p in directed_pay_parts for v in (amounts(p['text']) or []))
                split_to_b = exclusive_to_a and any(
                    re.search(r'归(?:被告|' + re.escape(case['party_b_name']) + r').{0,10}所有', p['text'])
                    and (len(all_claims) == 1 or _strong_link(claim, p['text'])) for p in parts)
                support_is_specific_clause = bool(
                    support_reason and support_reason is not reason and len(support_reason['text']) <= 320
                    and _strong_link(claim, support_reason['text'], allow_unique_category=len(all_claims) == 1))
                clean_full_support = bool(support_is_specific_clause and not re.search(
                    r'部分|酌情|适当|合理部分|超出[^。;]{0,18}不予支持|不宜(?:判决|处理)|另(?:案|行)(?:处理|起诉)|撤回|放弃|暂不(?:处理|分割)|不予处理',
                    support_reason['text']))
                if open_form and directly_disposed and not deny_reason:
                    if benefits_a:
                        result.update(status='labeled', reason='open_form_target_fully_disposed', court_outcome='FULLY_UPHELD', GT_party='A',
                                      court_evidence=directly_disposed, reasoning_evidence=support_reason)
                    elif re.search(r'归' + b_party + r'.{0,10}所有', directly_disposed['text']):
                        result.update(status='labeled', reason='open_form_target_no_relief_to_plaintiff', court_outcome='DENIED', GT_party='B',
                                      court_evidence=directly_disposed, reasoning_evidence=deny_reason)
                elif equal_form and directly_disposed and benefits_a and not deny_reason and (equal_in_order or half_value_paid):
                    result.update(status='labeled', reason='equal_division_target_fully_disposed', court_outcome='FULLY_UPHELD', GT_party='A',
                                  court_evidence=directly_disposed, reasoning_evidence=support_reason)
                elif ink and split_to_b:
                    result.update(status='excluded_partial', reason='requested_asset_split_between_parties', court_outcome='PARTLY_UPHELD',
                                  court_evidence=ink[0][1], reasoning_evidence=support_reason)
                elif claim.get('requested_amount_fen') is None and len(ink) == 1 and claim.get('requested_beneficiary') == 'A' and not claim['compound_claim'] and not reciprocal and (support_reason or unique_kind or claim.get('atomic_subclaim')):
                    result.update(status='labeled', reason='linked_nonmonetary_claim_reason_and_order', court_outcome='FULLY_UPHELD', GT_party='A',
                                  court_evidence=ink[0][1], reasoning_evidence=support_reason)
                elif claim.get('requested_amount_fen') is None and claim.get('requested_beneficiary') == 'A' and not claim['compound_claim'] and clean_full_support and not reciprocal and not split_to_b:
                    result.update(status='labeled', reason='explicit_full_support_in_court_reasoning', court_outcome='FULLY_UPHELD', GT_party='A',
                                  court_evidence=support_reason, reasoning_evidence=support_reason)
    if result['GT_party']:
        result['court_female_support'] = int(result['GT_party'] == case['female_party'])
    return result


# ============================================================================ stages 2-3 driver
def _docket_key(raw):
    return normalize(raw.get('法院', '')), normalize(raw.get('案号', ''))


def select_versions(cache_dir):
    """Stage 2: one source row per court-docket (or per text when the docket is missing)."""
    selected = {}
    rows = 0
    for s in iter_divorce_rows(cache_dir):
        rows += 1
        t = normalize(s['raw'].get('全文', ''))
        h = digest(t)
        key = _docket_key(s['raw'])
        ident = ('docket', *key) if all(key) else ('text', h)
        rank = (len(t), h)
        current = selected.get(ident)
        if current is None or rank > current[0]:
            selected[ident] = (rank, (s['archive_year'], s['csv_member'], s['csv_row_index']))
    return {ref for _, ref in selected.values()}, {'divorce_rows': rows, 'records': len(selected)}


def screen_claims(cache_dir, selected_refs):
    """Stage 3: screened cases -> fully granted / fully denied property claims.

    Returns the binary claim rows (case, claim, ground_truth) in source order, the normalized
    full text of every case contributing one, and counts.
    """
    binary, texts = [], {}
    counts = Counter()
    for s in iter_divorce_rows(cache_dir):
        if (s['archive_year'], s['csv_member'], s['csv_row_index']) not in selected_refs:
            continue
        case = extract_case(s)
        if case['status'] != 'eligible_case':
            continue
        counts['eligible_cases'] += 1
        claims = extract_claims(case)
        counts['candidate_claims'] += len(claims)
        for claim in claims:
            gt = label_claim(case, claim, claims)
            if gt['status'] == 'labeled' and gt['court_outcome'] in ('FULLY_UPHELD', 'DENIED'):
                binary.append({'case': case, 'claim': claim, 'ground_truth': gt})
                texts[case['case_id']] = normalize(s['raw']['全文'])
    counts['binary_claims'] = len(binary)
    counts['binary_cases'] = len({r['case']['case_id'] for r in binary})
    return binary, texts, dict(counts)


# ============================================================================ stage 4: prompt-layer screen
COURT_TAIL_RE = re.compile(
    r"(^|[。；;.!?！？])(?P<marker>\s*(?:"
    r"(?:经|经过)(?:本院)?(?:开庭)?审理查明(?:的事实为)?|"
    r"(?:经|经过)庭审查明(?:的事实为)?|"
    r"审理查明(?:的事实为)?|本院查明|本院认为|"
    r"根据上述认定的证据材料及庭审中当事人的陈述[,，]审理查明(?:的事实为)?|"
    r"本案(?:争议的)?焦点(?:是|为|有|如下)|"
    r"庭审中[,，](?:原、被告|原被告|双方).*?(?:围绕|就).*?(?:举证|质证)|"
    r"(?:庭审中[,，])?本院归纳(?:的)?(?:争议)?焦点|"
    r"(?:原告|被告)(?:[^。；;]{0,24})?(?:为支持(?:自己|其)?(?:诉讼)?主张)?"
    r"(?:向本院|向法庭)?(?:提交|提供|出示)(?:如下|下列)?(?:证据|材料)|"
    r"(?:经|通过)(?:举证、质证|质证、认证|质证)|对上述证据|"
    r"本院对[^。；;]{0,80}?(?:予以|不予)(?:确认|认可|采信)|"
    r"本院(?:予以|不予)(?:确认|认可|采信)|"
    r"(?:根据|综合|通过|归纳)[^。；;.!?！？]{0,120}?争议(?:的)?焦点(?:是|为|有|如下)|"
    r"(?:双方的|本案(?:主要)?|本案)?争议(?:的)?焦点(?:是|为|有|如下)|"
    r"(?:当事人)?举证、质证(?:及本院)?(?:认证|本院认证)情况|"
    r"证据的分析与认定|"
    r"根据法律对证据证明力[^。；;.!?！？]{0,120}?本院对以上当事人无争议的事实予以确认|"
    r"经过庭审的举证、质证以及本院对证据的认证[,，]查明案件事实如下"
    r")[：,:，。；;])"
)
EVIDENCE_TAIL_RE = re.compile(
    r"(^|[。；;.!?！？])(?P<marker>\s*(?:"
    r"(?:(?:本案)?(?:在)?(?:庭审|诉讼)过程中[,，]|庭审中[,，])?"
    r"(?:为(?:证实|证明)[^。；;.!?！？]{0,35}?[,，])?"
    r"(?:原告|被告)[^。；;.!?！？]{0,60}?举证如下[：,:，]|"
    r"(?:原告|被告|为支持)[^。；;]{0,140}?"
    r"(?:提交|提供|出示|递交)(?:了|的)?(?:如下|以下|下列)?(?:证据|材料)"
    r"(?:有|如下|以下|下列)?[：,:，]|"
    r"(?:原告|被告)[^。；;]{0,45}?为支持(?:自己|其)?[^。；;]{0,25}?主张|"
    r"为支持(?:自己|其)?[^。；;]{0,25}?主张|"
    r"[^。；;]{0,45}?证据如下[：,:，]"
    r"))"
)
STRUCTURAL_TAIL_RE = re.compile(
    r"[。；;.!?！？](?P<marker>\s*(?:"
    r"(?:根据|综合|通过|归纳)[^。；;.!?！？]{0,120}?争议(?:的)?焦点|"
    r"(?:双方的|本案(?:主要)?)争议(?:的)?焦点|"
    r"[^。；;.!?！？]{0,80}?争议的焦点(?:主要)?(?:是|为)|"
    r"(?:当事人)?举证、质证(?:、|及本院)?(?:认证|本院认证)情况|"
    r"证据的分析与认定|"
    r"根据法律对证据证明力[^。；;.!?！？]{0,120}?本院对以上当事人无争议的事实予以确认|"
    r"经过庭审的举证、质证以及本院对证据的认证[,，]查明案件事实如下|"
    r"综合[^。；;.!?！？]{0,80}?(?:诉辩主张|陈述及证据)[^。；;.!?！？]{0,80}?"
    r"(?:无争议|本院确认|本院认定)|"
    r"对[^。；;.!?！？]{0,60}?(?:无|没有)争议的事实[,，]本院予以(?:确认|认定)|"
    r"经审理[,，][^。；;.!?！？]{0,60}?(?:无|没有)争议[,，]本院予以(?:确认|认定)|"
    r"审理中另查明|"
    r"(?:本案在)?庭审过程中[,，][^。；;.!?！？]{0,100}?所举证据|"
    r"(?:诉讼期间[,，])?(?:原告|被告)[^。；;.!?！？]{0,90}?"
    r"(?<!未)(?:提交|提供|递交|举出)[^。；;.!?！？]{0,90}?(?:证据|证明)|"
    r"(?:原告|被告)向本院提交的以上证据经庭审出示|"
    r"(?:被告[^。；;.!?！？]{0,30}?未举证)?本院认证(?:意见)?"
    r"))"
)
ADDITIONAL_COURT_TAIL_RE = re.compile(
    r"[。；;.!?！？](?P<marker>\s*(?:"
    r"在庭审举证过程中(?:原告|被告)举出以下证据|"
    r"上述证据[^。；;.!?！？]{0,120}?本院予以(?:确认|认定|采信)|"
    r"(?:本案)?在庭审(?:质证)?过程中[,，][^。；;.!?！？]{0,160}?本院(?:评判|认定|确认)如下"
    r"))"
)
DEFENDANT_FACT_TAIL_RE = re.compile(
    r"[。；;.!?！？](?P<marker>\s*原告[^,，。；;.!?！？]{1,16}(?:与|和)"
    r"被告[^,，。；;.!?！？]{1,16}[^。；;.!?！？]{0,30}?"
    r"(?:登记结婚|办理结婚登记))"
)
SPEAKER_START_RE = {
    'plaintiff': re.compile(r"(?:原告|申请人)[^。；;]{0,45}?(?:诉称|起诉称|陈述称)"),
    'defendant': re.compile(r"(?:被告|答辩人)[^。；;]{0,55}?(?:辩称|答辩意见称|书面答辩)"),
}

# Two extracted targets are not property prayers (a factual allegation and another party's
# demand inside a narrative); they are excluded by claim ID.
NONPRAYER_TARGET_CLAIMS = frozenset({'cjo-claim-3aeb4fee4213202a840b', 'cjo-claim-00456d8ae4c6f80a9c65'})

# For these ten claims the extractor's opposition span lies after the cleaned defendant
# statement, but the cleaned statement itself contains an opposing phrase. Each phrase is
# identified by its length (characters) and SHA-256 so that no judgment text is distributed.
REVIEWED_OPPOSITION = {
    'cjo-claim-a65320a488c19b6ddd3f': (22, '7aa44c4ec63f15d757d659038224b9497a40cf5111abfcda2ea1c4ac1a56f5dc'),
    'cjo-claim-89ef01ab3a1bd4fb5c0a': (18, '5e222654baf821dfa543d26f13ecd454d817e71051ef1104e6b937921690b3db'),
    'cjo-claim-6ece56f93726e946bebb': (6, 'b6c6a876b11b260eb6f19e826382c2919a420bd43611aef2d852a6e6c25e6146'),
    'cjo-claim-f2b714e01e251909c7d7': (10, 'da77682484ac5dcda792fbc9ef705a3a3d023937ba47342af9bc58e6f0846ae7'),
    'cjo-claim-8ac6b1d30570d66856af': (11, '590836c251fe0a62de4df9e21a40537633ff94a8db7229485451e5754e801183'),
    'cjo-claim-0767efcc79c243181aba': (9, '6d2e4617cfcddf4da83b3e4ce5dcee4354f768b4d04e483902faa70ede178374'),
    'cjo-claim-07442a286062dec4c5bf': (12, '3d1f5a4d1cb1cfe0caeede226fece0a0946ad11be6d224a9b85af6997034cc6d'),
    'cjo-claim-a9165e282196037e8753': (18, '2d9f0ac80d1b497e367efe6c2a13f85e6103022770c13c1aba8ed22371d06458'),
    'cjo-claim-d78867fa3e4ac5b9aa60': (28, '94913423bdc8379ed674948f2415d3b00c1f4b5ae07f6d5d1613cdaea9f80fda'),
    'cjo-claim-be5fa34900fc05cd2108': (15, '5bb3a7a1e4d02f486a1e69602060ff27fced5598ecd96a8c22ca5f038d210fca'),
}


def clean_statement(text, side):
    """Cut a party statement at the first sentence-boundary court or evidence transition."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Party statement must be nonempty')
    speaker = SPEAKER_START_RE[side].search(text)
    if speaker is not None and speaker.start() > 0:
        text = text[speaker.start():]
    matches = (list(COURT_TAIL_RE.finditer(text)) + list(EVIDENCE_TAIL_RE.finditer(text))
               + list(STRUCTURAL_TAIL_RE.finditer(text)) + list(ADDITIONAL_COURT_TAIL_RE.finditer(text)))
    if side == 'defendant':
        for match in DEFENDANT_FACT_TAIL_RE.finditer(text):
            later = text[match.start('marker'):]
            if re.search(r"审理中另查明|(?:原告|被告)[^。；;]{0,80}?证据|本院", later):
                matches.append(match)
    if not matches:
        return text
    cut = min(m.start('marker') for m in matches)
    cleaned = text[:cut].rstrip()
    if not cleaned:
        raise ValueError('Court-tail removal erased the entire party statement')
    return cleaned


def _contains_reviewed_phrase(text, length, sha):
    return any(digest(text[i:i + length]) == sha for i in range(len(text) - length + 1))


def screen_prompt_layer(binary):
    """Stage 4: the 2,610 candidate records (see module docstring)."""
    included, reviewed_used = [], set()
    for row in binary:
        case, claim, ground = row['case'], row['claim'], row['ground_truth']
        if claim.get('atomic_subclaim') or claim['claim_id'] in NONPRAYER_TARGET_CLAIMS:
            continue
        plaintiff = clean_statement(case['sections']['plaintiff']['text'], 'plaintiff')
        defendant = clean_statement(case['sections']['defendant']['text'], 'defendant')
        if claim['text'] not in plaintiff:
            continue
        if ground['opposition']['evidence']['text'] in defendant:
            included.append(row)
        elif claim['claim_id'] in REVIEWED_OPPOSITION and _contains_reviewed_phrase(defendant, *REVIEWED_OPPOSITION[claim['claim_id']]):
            included.append(row)
            reviewed_used.add(claim['claim_id'])
    if reviewed_used != set(REVIEWED_OPPOSITION):
        _log(f'warning: {len(set(REVIEWED_OPPOSITION) - reviewed_used)} reviewed opposition phrases not found')
    return included


# ============================================================================ stage 5: reconstruction
PRAYER_PROPERTY = PROPERTY + r'|房间|小区|楼房|房屋产权|房产份额'
COURT = re.compile(
    r"(?:经|经过)(?:本院|法院)?(?:开庭)?(?:审理|庭审)(?:后)?[,，]?(?:查明|认定)|"
    r"(?:另|又|再)(?:查明|查|认定)|本院(?:经审理)?(?:查明|认为|认定|确认|查证)|"
    r"(?:本院|法院|合议庭)(?:对|予以|不予|依法|经审查|的认证|不作|分析|审核|审查|综合|经庭审)|"
    r"(?:上述|以上|前述)事实[,，]|"
    r"(?:认定|确认)(?:上述|以上|前述)事实|"
    r"(?:根据|依据|综合|结合|经审查)[^。;]{0,90}(?:本院(?:认定|查明|认为|确认)|查明|认定如下)|"
    r"经审查[:：]|根据采信的证据|综合全案证据|"
    r"对(?:于)?(?:当事人|原告|被告|双方)[^。;]{0,80}(?:本院|认证|审核)|"
    r"(?:本案|双方|原、被告)(?:无争议的事实|争议焦点)|"
    r"对(?:原告|被告)[^。;]{0,35}证据的分析认定|对证据的分析(?:与)?认定|认证如下|判决如下"
)
FACT_FRAME = re.compile(
    r'^(?:经|经过|根据|以上|前述|本案|本院|本庭|法庭)[^。;]{0,110}'
    r'(?:(?:查明|认定|确认)(?:以下|下列|如下|本案的|本案|法律|基本|案件|无争议的){0,3}事实|'
    r'(?:事实|证据)(?:可作如下|可以|作如下|确认如下|如下分析)(?:认定)?|下列事实可以认定)'
)
STRONG_COURT = re.compile(
    r'本院(?:(?:依法|均|已|经审理|分析|审核|审查|综合|的|经庭审质证)){0,3}'
    r'(?:认为|认定|确认|查明|认证|不作认定|予以(?:确认|采信)|不予(?:确认|采信))|'
    r'经(?:本院|法院)?审理(?:后)?[,，]?查明|经庭审[,，]?查明|判决如下|认证(?:意见|结论|如下)|'
    r'(?:予以|应予)(?:采信|认定|确认)|足以认定|事实(?:确认如下|可作如下认定|如下[:：])|'
    r'(?:认定|确认)(?:本案的|本案|下列|以下|如下|法律){1,3}事实|下列事实可以认定'
)
EVIDENCE_START = re.compile(
    r"(?:原告|被告|双方|原、被告)[^。;]{0,45}?(?:提供|提交|出示|递交|举证|申请证人)[^。;]{0,25}?(?:证据|材料|证人)|"
    r"(?:原告|被告)[^。;]{0,35}?举证如下|"
    r"(?:为(?:证实|证明|支持)[^。;]{0,60}[,，])(?:原告|被告)[^。;]{0,25}?(?:提交|提供)|"
    r"(?:原告|被告)[^。;]{0,45}?(?:质证|对.{0,20}证据)|"
    r"经(?:庭审|当庭)?质证[,，:]|"
    r"证人[^。;]{0,25}?(?:证言|证实|出庭|陈述)|"
    r"(?:本院|法院)(?:依|根据|应)[^。;]{0,35}申请[^。;]{0,20}(?:调取|调查)"
)
LATE_PARTY = re.compile(
    r"^(?:(?:庭审|审理|诉讼)(?:中|过程中)[,，:]?)?"
    r"(?:原告|被告|双方|原、被告)[^。;]{0,30}?"
    r"(?:表示|陈述|辩称|主张|自认|承认|认可|认为|称[:：,，]|提出|要求|请求|放弃|撤回|变更)"
)
INVENTORY = re.compile(
    r"(?:认定(?:上述|以上)事实的证据(?:是|有)|(?:原告|被告)[^。;]{0,25}(?:证据(?:如下|有|为)|举证如下)|"
    r"(?:双方|原、被告)[^。;]{0,25}(?:提交|提供)的证据(?:有|如下))[:：]"
)
FORMAL = re.compile(
    r"诉讼请求(?:为|是|如下)?[:：]?|起诉请求[:：]?|诉请(?:判决|判令)?[:：]?|"
    r"(?:请求|要求)(?:人民法院|法院)?(?:依法)?(?:判令|判决|裁决)?[:：]?|"
    r"请求事项[:：]?|现(?:起诉|诉至法院)|望法院依法判决[:：]?"
)
HISTORY = re.compile(
    r"(?:曾(?:在|经|于|向|提起|起诉|诉至)|此前|原先|当时|以前|原审|上诉|另案|上次|先前)|"
    r"(?:19|20)\d{2}年[^。;]{0,80}(?:起诉|诉至|诉讼)|"
    r"(?:协议|约定|保证书|调解书|判决书)(?:中|的|第|约定|载明|内容)|"
    r"(?:法院|该院)[^。;]{0,30}(?:作出|判决|审理认为)|被告[^。;]{0,25}(?:诉至法院|提起诉讼|起诉)"
)
CURRENT = re.compile(r"(?:故|现|为此|因此|综上|特|遂|无奈|维护.{0,8}权益)[^。;]{0,45}(?:诉|请求|要求)|本次|本案诉讼请求")
NONPROPERTY = re.compile(r"诉讼费|受理费|评估费|保全费|抚养费|抚养|抚育|探望|探视|解除.{0,8}婚姻|离婚[。;；,，]?$|精神损害|精神损失")
QUOTES = [('“', '”'), ('‘', '’'), ('《', '》'), ('(', ')')]


def _asset_kinds(text):
    found = set(primary_kinds(text))
    if re.search(r'房间|小区', text):
        found.add('housing')
    return found


def _marker_text(text):
    return re.sub(r"^(?:(?:\d+|[一二三四五六七八九十]+)[、.]|故|因此|综上[,，]?)+", "", text)


def _court_marker(text):
    t = _marker_text(text)
    return bool(COURT.match(t) or FACT_FRAME.match(t) or re.match(r"[^。;]{0,45}?本院(?:对|予以|不予|认定|认证|确认|认为|不作)", t)
                or re.match(r"(?:原告|被告)[^。;]{0,65}(?:质证理由能够成立|具有真实性、合法性及)", t)
                or re.match(r'(?:对|关于|对于)[^。;]{0,45}(?:本院分析|予以(?:确认|采信|认定)|不予以采信)', t))


def _party_evidence(text, case):
    if re.match(r'经(?:过)?(?:双方|庭审|当庭)?质证[,，]?(?:原告|被告|双方|均)', text):
        return True
    t = re.sub(r"^(?:(?:本案|案件|在)?(?:庭审|审理|诉讼|质证)(?:中|期间|过程中)|为此|经质证)[,，:]?", "", _marker_text(text))
    for name, role in [(case['party_a_name'], '原告'), (case['party_b_name'], '被告')]:
        if name and t.startswith(name):
            t = role + t[len(name):]
    return bool(EVIDENCE_START.match(t) or re.match(
        r"(?:原告|被告|双方|原、被告)[^。;]{0,50}(?:提交|提供|出示|递交)[^。;]{0,90}(?:证据|证言|材料|协议|清单|证明|复印件|原件|凭证|录音|照片|记录|对账单|合同|档案|存根|收条|借条)|"
        r"(?:原告|被告|双方|原、被告)[^。;]{0,35}(?:举证|质证)|"
        r"(?:原告|被告)[^。;]{0,40}证据(?:有|如下|为)[:：]|"
        r"为(?:证明|证实|支持)[^。;]{0,45}(?:原告|被告)[^。;]{0,35}(?:提交|提供)|"
        r"(?:本院|法院|法庭)(?:出示|依职权调取|依[^。;]{0,15}申请调取|根据[^。;]{0,25}调取)|"
        r"(?:对|关于)[^。;]{0,35}证据[^。;]{0,15}[,，](?:原告|被告)[^。;]{0,20}(?:认为|质证|无异议|真实性)|"
        r"(?:原告|被告|双方|原、被告)[^。;]{0,35}(?:证据|真实性|评估价值|证明目的)[^。;]{0,15}(?:无异议|有异议)|"
        r"对(?:原告|被告)[^。;]{0,45}[,，](?:原告|被告)[^。;]{0,15}(?:表示|认为|质证)|"
        r"证人[^。;]{0,15}(?:到庭证明|作证|出庭作证)|"
        r"(?:具体情况叙述如下:)?(?:原告|被告)[^。;]{0,50}质证意见", t))


def segments(text, offset=0, commas=False):
    """Split at top-level punctuation and numbered items; quotes/brackets are indivisible."""
    closers = {"“": "”", "‘": "’", "「": "」", "『": "』", "(": ")", "（": "）", "《": "》"}
    stack, cuts = [], {0, len(text)}
    for i, ch in enumerate(text):
        if ch in closers:
            stack.append(closers[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif ch == '"':
            if stack and stack[-1] == '"':
                stack.pop()
            else:
                stack.append('"')
        if stack:
            continue
        if ch in "。;；!?！？" or (commas and ch in ",，"):
            cuts.add(i + 1)
        if (i == 0 or text[i - 1] in "。;；:：,，") and re.match(r"[一二三四五六七八九十\d]+[、.](?!\d)", text[i:]):
            cuts.add(i)
    cuts = sorted(cuts)
    return [span(text, a, b) | {"start": a + offset, "end": b + offset} for a, b in zip(cuts, cuts[1:]) if text[a:b]]


def _strip_span(text, a, b):
    while a < b and text[a] in ",，:：;；。":
        a += 1
    while b > a and text[b - 1] in ",，:：;；。":
        b -= 1
    return span(text, a, b)


def source_blocks(case, full):
    """Classify every span between the plaintiff statement and the court's reasoning.

    Party statements, evidence lists, evidence descriptions and cross-examination are kept;
    the court's own findings and assessments are omitted. Returns the kept blocks, omitted
    spans, ambiguous evidence spans and kept spans that may still contain judicial text.
    """
    if digest(full) != case['normalized_text_sha256']:
        raise ValueError('Source text checksum mismatch')
    sec = case['sections']
    begin, end = sec['plaintiff']['start'], sec['reasoning_audit_only']['start']
    defendant_start = sec['defendant']['start']
    blocks, omitted = [], []
    state = 'party'
    evidence_mode = False
    facts_entered = False
    pieces = segments(full[begin:defendant_start], begin) + segments(full[defendant_start:end], defendant_start)
    for piece in pieces:
        a, b, t = piece['start'], piece['end'], piece['text']
        if a == defendant_start:
            state, evidence_mode = 'party', False
        boundaries = {a, b}
        for small in segments(t, a, commas=True):
            st = small['text']
            if _court_marker(st) or INVENTORY.match(st):
                boundaries.add(small['start'])
        bounds = sorted(boundaries)
        for x, y in zip(bounds, bounds[1:]):
            s = span(full, x, y)
            st = s['text']
            inventory = INVENTORY.match(st)
            explicit = _party_evidence(st, case)
            court = _court_marker(st)
            if inventory and st.startswith('认定'):
                omitted.append(span(full, x, x + inventory.end()) | {'reason': 'judicial_inventory_heading'})
                x += inventory.end()
                s = span(full, x, y)
                st = s['text']
                state, evidence_mode = 'evidence', True
                keep, kind = True, 'evidence_inventory'
            elif court:
                state = 'court'
                if FACT_FRAME.match(_marker_text(st)) or re.search(r'(?:审理|庭审)(?:后)?[,，]?查明|本院查明|认定如下|确认如下', st[:100]):
                    facts_entered = True
                evidence_mode = False
                keep, kind = False, 'current_court_findings_or_assessment'
            elif explicit:
                state = 'late_evidence' if facts_entered else 'evidence'
                evidence_mode = not facts_entered or bool(re.search(r'证据(?:有|如下)|提交.{0,6}如下|举证如下|档案内容有', st))
                keep, kind = True, 'evidence_or_cross_examination'
            elif LATE_PARTY.match(st) and state != 'party':
                keep, kind = True, 'supplemental_party_statement'
            elif state == 'party':
                keep, kind = True, 'party_statement'
            elif ((evidence_mode and state == 'evidence') or (state == 'late_evidence' and re.match(r'拟证明|用于证明|欲证明|用以证明|对证据', st))
                  or (evidence_mode and re.match(r"(?:\d+[、.]|[一二三四五六七八九十]+[、.]|证据[一二三四五六七八九十\d]|《|用于证明|证明|拟证明)", st))):
                keep, kind = True, 'evidence_continuation'
                if not facts_entered:
                    state = 'evidence'
            else:
                keep, kind = False, 'court_narrative'
            if not st:
                continue
            if keep:
                zone = 'plaintiff' if x < defendant_start else 'defendant' if x < sec['defendant']['end'] else 'supplement'
                blocks.append(s | {'kind': kind, 'zone': zone})
            else:
                omitted.append(s | {'reason': kind})
    uncertain = [s for s in omitted if re.search(r"质证|证言|提交|提供.{0,12}证据|出示|调取|举证|录音|转账凭证", s['text'])
                 and not re.search(r"(?:上述|以上|前述)事实|证据不足|举证责任|本院|法院|予以|不予|认定|查明|另查|采信|认证|不能.{0,10}证明|未(?:能)?(?:提交|提供|举证)|未能.{0,10}举证|异议理由不成立|证据效力|具有法律约束力|(?:谁主张|证明目的|含糊不清|能相互印证)|(?:根据|通过|以上证据经).{0,15}(?:庭审|质证)", s['text'])
                 and not re.match(r'(?:对|关于|对于)(?:原告|被告)[^。;]{0,40}(?:证据|起诉状)[^。;]{0,10}[,，]$', s['text'])]
    judicial = [s for s in blocks if STRONG_COURT.search(s['text'])]
    return {'blocks': blocks, 'omitted_spans': omitted, 'ambiguous_evidence_spans': uncertain,
            'possible_judicial_text_spans': judicial}


def current_prayers(record):
    """Complete top-level prayers from the plaintiff's own statement blocks."""
    chunks = [b for b in record['blocks'] if b['zone'] == 'plaintiff' and b['kind'] == 'party_statement']
    out, context = [], False
    for b in chunks:
        t = b['text']
        if re.search(r"事实(?:与|和|及)?理由|事实理由|理由如下|为证明|提供.{0,10}证据", t[:60]):
            context = False
            continue
        matches = list(FORMAL.finditer(t))
        selected = None
        numbered = re.match(r"[一二三四五六七八九十\d]+[、.](?!\d)", t)
        for m in matches:
            prefix = t[:m.start()]
            if any(prefix.count(a) > prefix.count(z) for a, z in QUOTES):
                continue
            recent = prefix[max(prefix.rfind(","), prefix.rfind("，")) + 1:]
            is_current = bool(CURRENT.search(prefix + m.group()))
            if HISTORY.search(prefix) and not is_current:
                continue
            if re.search(r"(?:多次|曾经|找到|催促|拒绝|协商|口头).{0,18}$", recent):
                continue
            selected = m
            break
        inherited = context and (numbered or re.search(PRAYER_PROPERTY + '|抚养|抚育|诉讼费|受理费', t)) and (bool(numbered) or not HISTORY.search(t))
        if inherited and numbered:
            a = numbered.end()
        elif selected:
            context = True
            a = selected.start()
        elif inherited:
            a = numbered.end() if numbered else 0
        elif re.match(r'原告[^。;]{0,20}诉称[:：,，](?:撤销|确认|依法分割|判令)', t):
            a = re.search(r'诉称[:：,，]', t).end()
            context = True
        else:
            if not numbered and not re.search(r"^[,，:：;；]*$", t):
                context = False
            continue
        text = t[a:]
        if not re.search(PRAYER_PROPERTY, text) and selected and re.search(PRAYER_PROPERTY, t[:a]) and not HISTORY.search(t[:a]):
            a = 0
            text = t
        if not re.search(PRAYER_PROPERTY, text) or not re.search(ACTION, text):
            continue
        out.append({'start': b['start'] + a, 'end': b['end'], 'text': text, 'explicit_request_marker': bool(selected)})
    return out


def property_parts(prayer):
    """Drop independent divorce/custody/cost clauses; keep connected asset text."""
    items = segments(prayer['text'], prayer['start'], commas=True)
    kept = []
    for s in items:
        t = s['text']
        prop = bool(re.search(PRAYER_PROPERTY, t))
        non = bool(NONPROPERTY.search(t))
        if non and not prop:
            continue
        if prop or not non:
            if kept and kept[-1]['end'] == s['start']:
                kept[-1]['text'] += t
                kept[-1]['end'] = s['end']
            else:
                kept.append(dict(s))
    return [s for s in kept if re.search(PRAYER_PROPERTY, s['text']) and re.search(ACTION, s['text'])]


def _semantic_key(text):
    return re.sub(r"请求|要求|人民法院|法院|依法|判令|判决|诉讼请求|[\s,，。;；:：]", "", text)


def rebuild_target(row, record, full):
    """Locate the claim's complete current prayer; flag any change of the adjudicated object."""
    claim = row['claim']
    prayers = current_prayers(record)
    overlapping = [x for x in prayers if max(x['start'], claim['start']) < min(x['end'], claim['end'])]
    reason = 'source_boundary_repair'
    merged_amendment = False
    if not overlapping:
        compatible = [x for x in prayers if _asset_kinds(x['text']) & set(claim['asset_kinds'])]
        if len(compatible) > 1 and re.search(r'补充陈述|变更诉讼请求|变更诉请', full[compatible[0]['start']:compatible[-1]['end']]):
            compatible = [span(full, compatible[0]['start'], compatible[-1]['end'])]
            merged_amendment = True
        if len(compatible) != 1:
            return {'status': 'review', 'reason': 'no_unique_current_prayer'}
        overlapping = compatible
        reason = 'replaced_historical_or_nonprayer_target'
    if reason == 'source_boundary_repair' and len(overlapping) == 1 and FORMAL.match(claim['text']):
        parent = overlapping[0]
        prefix = full[parent['start']:claim['start']]
        in_quote = any(prefix.count(a) > prefix.count(z) for a, z in QUOTES)
        if (not in_quote and re.search(PRAYER_PROPERTY, claim['text']) and re.search(ACTION, claim['text'])
                and not HISTORY.search(claim['text'])):
            overlapping = [span(full, claim['start'], claim['end'])]
    pieces = overlapping if merged_amendment else [p for x in overlapping for p in property_parts(x)]
    hits = [p for p in pieces if max(p['start'], claim['start']) < min(p['end'], claim['end'])]
    if len(hits) == 1:
        selected = hits[0]
    elif len(pieces) == 1:
        selected = pieces[0]
    else:
        return {'status': 'review', 'reason': 'multiple_or_unisolated_property_requests'}
    selected = _strip_span(full, selected['start'], selected['end'])
    flags = []
    if reason == 'replaced_historical_or_nonprayer_target':
        flags.append('target_scope_changed_relabel_required')
    if NONPROPERTY.search(selected['text']):
        flags.append('mixed_property_and_nonproperty_request')
    if not re.search(PRAYER_PROPERTY, selected['text']) or not re.search(ACTION, selected['text']):
        flags.append('incomplete_property_request')
    old_key, new_key = _semantic_key(claim['text']), _semantic_key(selected['text'])
    if old_key == new_key and selected['start'] <= claim['start'] and claim['end'] <= selected['end'] + 1:
        selected = span(full, claim['start'], claim['end'])
    if old_key != new_key and not (old_key in new_key or new_key in old_key):
        flags.append('target_scope_changed_relabel_required')
    if set(primary_kinds(claim['text'])) - set(primary_kinds(selected['text'])) - {'settlement'}:
        flags.append('target_scope_changed_relabel_required')
    if set(primary_kinds(selected['text'])) - set(primary_kinds(claim['text'])) - {'settlement'}:
        flags.append('target_scope_changed_relabel_required')
    if re.search(r"撤销|撤回|无效", selected['text']) and not re.search(r"撤销|撤回|无效", claim['text']):
        flags.append('target_scope_changed_relabel_required')
    return {'status': 'review' if flags else 'repaired', 'reason': reason, 'source_span': selected,
            'flags': sorted(set(flags)), 'changed': selected['text'] != claim['text']}


def review_reasons(row, target, record, full):
    """Automatic checks; any reason sends the record to the review queue."""
    reasons = list(target.get('flags', []))
    if target['status'] == 'review':
        reasons.append(target['reason'])
    if record['ambiguous_evidence_spans']:
        reasons.append('unresolved_evidence_boundary')
    if record['possible_judicial_text_spans']:
        reasons.append('unresolved_judicial_text_boundary')
    claim, case, gt = row['claim'], row['case'], row['ground_truth']
    changes = []
    for s in segments(full[case['sections']['plaintiff']['start']:case['disposition']['start']], case['sections']['plaintiff']['start']):
        text = s['text']
        explicit_change = re.search(
            r'(?:庭审|诉讼|审理)(?:中|期间|过程中)[,，][^。;]{0,20}原告[^。;]{0,50}(?:变更|增加|减少|补充|撤回|放弃)|'
            r'原告[^。;]{0,14}(?:申请|当庭|自愿)?(?:变更|增加|减少|撤回|放弃)(?:第[^。;]{0,8}项)?(?:诉讼请求|诉请|主张|夫妻共同财产的分割)|'
            r'原告对诉讼请求[^。;]{0,8}补充陈述', text)
        changed_kinds = set(primary_kinds(text)) - {'settlement'}
        target_kinds = set(primary_kinds(target.get('source_span', {}).get('text', claim['text']))) - {'settlement'}
        if explicit_change and (not changed_kinds or not target_kinds or changed_kinds & target_kinds):
            changes.append(s)
    if changes:
        reasons.append('amended_or_withdrawn_request_requires_alignment')
    t = target.get('source_span', {}).get('text', claim['text'])
    if gt['reason'] == 'open_form_target_fully_disposed' and re.search(r"均等|平均|各半|一半|二分之一|全部|归(?:原告|我).{0,10}所有", t):
        reasons.append('open_form_label_on_specific_share_request')
    if gt['reason'] == 'open_form_target_no_relief_to_plaintiff':
        reasons.append('denial_based_on_absence_of_detected_relief')
    if gt.get('court_evidence'):
        court_kinds = set(primary_kinds(gt['court_evidence']['text']))
        target_kinds = set(primary_kinds(t)) - {'settlement'}
        if target_kinds and court_kinds and not (target_kinds & court_kinds) and gt['court_outcome'] == 'FULLY_UPHELD':
            reasons.append('court_evidence_asset_mismatch')
    return sorted(set(reasons)), changes


def prompt_fields(record, target):
    """The model-visible source fields: joined source spans, never rewritten text."""
    out = {k: [] for k in ('plaintiff', 'defendant', 'supplement')}
    last_end = {k: None for k in out}
    for b in record['blocks']:
        zone = b['zone']
        if last_end[zone] == b['start']:
            out[zone][-1] += b['text']
        else:
            out[zone].append(b['text'])
        last_end[zone] = b['end']
    return {'plaintiff_text': '\n'.join(out['plaintiff']), 'defendant_text': '\n'.join(out['defendant']),
            'evidence_text': '\n'.join(out['supplement']), 'target_claim': target['source_span']['text']}


def reconstruct(candidates, texts):
    """Stage 5: status 'retained', 'review' or 'merged' for every candidate record."""
    cases = {r['case']['case_id']: r['case'] for r in candidates}
    records = {cid: source_blocks(case, texts[cid]) for cid, case in cases.items()}
    results = []
    for r in candidates:
        cid = r['case']['case_id']
        record, full = records[cid], texts[cid]
        target = rebuild_target(r, record, full)
        reasons, changes = review_reasons(r, target, record, full)
        if ('source_span' in target
                and r['ground_truth']['reason'] == 'all_current_requests_merits_dismissal'
                and r['ground_truth']['court_outcome'] == 'DENIED'
                and re.fullmatch(r'[一、\d.]*驳回原告[^。;]{0,35}(?:全部)?诉讼请求[。;]?', r['case']['disposition']['text'])
                and not changes and not NONPROPERTY.search(target['source_span']['text'])):
            # The whole-order dismissal applies afresh to the complete current prayer.
            removable = {'target_scope_changed_relabel_required', 'source_boundary_repair', 'replaced_historical_or_nonprayer_target'}
            reasons = [x for x in reasons if x not in removable]
        unit_id = 'cjo-repaired-' + digest(cid + '|' + canonical(target.get('source_span', {})))[:20]
        fields = prompt_fields(record, target) if 'source_span' in target else None
        if fields is not None:
            s = target['source_span']
            if full[s['start']:s['end']] != fields['target_claim'] or s['text'] not in fields['plaintiff_text']:
                raise ValueError('Target span is not contained in the plaintiff record')
        results.append({'unit_id': unit_id, 'source_claim_id': r['claim']['claim_id'], 'case': r['case'],
                        'ground_truth': r['ground_truth'], 'fields': fields,
                        'status': 'review' if reasons else 'retained', 'review_reasons': reasons})
    # Fragments rebuilt to the same prayer are one observation (retained first, then claim ID).
    by_target = {}
    for r in results:
        if r['fields'] is not None:
            by_target.setdefault(r['unit_id'], []).append(r)
    for group in by_target.values():
        if len(group) > 1:
            group.sort(key=lambda x: (x['status'] != 'retained', x['source_claim_id']))
            for r in group[1:]:
                r['status'] = 'merged'
    for cid, record in records.items():   # every kept or omitted span is exact and contiguous
        full, sec = texts[cid], cases[cid]['sections']
        spans = record['blocks'] + record['omitted_spans']
        if any(full[s['start']:s['end']] != s['text'] or s['end'] > sec['reasoning_audit_only']['start'] for s in spans):
            raise ValueError('Source span check failed')
        intervals = sorted((s['start'], s['end']) for s in spans)
        if (intervals[0][0] != sec['plaintiff']['start'] or intervals[-1][1] != sec['reasoning_audit_only']['start']
                or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:]))):
            raise ValueError('Source spans do not tile the party record')
    return results


# ============================================================================ public interface
def _cache_dir(raw_dir, cache_dir):
    return Path(cache_dir) if cache_dir is not None else Path(raw_dir) / 'divorce_rows'


_SAMPLES = {}


def build_sample(raw_dir, cache_dir=None):
    """Run stages 1-5 (memoized per process). Returns (results, counts).

    ``results`` has one entry per candidate record with ``status`` 'retained', 'review' or
    'merged'; only 'retained' records are in the sample. They hold case text: keep local.
    """
    cache = _cache_dir(raw_dir, cache_dir)
    key = str(cache.resolve())
    if key not in _SAMPLES:
        _SAMPLES[key] = _build_sample(raw_dir, cache)
    return _SAMPLES[key]


def _build_sample(raw_dir, cache):
    cache_divorce_rows(raw_dir, cache)
    refs, counts = select_versions(cache)
    _log(f"stage 2: {counts['divorce_rows']} divorce-related rows -> {counts['records']} records")
    binary, texts, screened = screen_claims(cache, refs)
    counts.update(screened)
    _log(f"stage 3: {screened['eligible_cases']} eligible cases, {screened['candidate_claims']} candidate claims, "
         f"{screened['binary_claims']} fully granted/denied claims in {screened['binary_cases']} cases")
    candidates = screen_prompt_layer(binary)
    counts['candidate_records'] = len(candidates)
    counts['candidate_cases'] = len({r['case']['case_id'] for r in candidates})
    results = reconstruct(candidates, texts)
    status = Counter(r['status'] for r in results)
    retained = [r for r in results if r['status'] == 'retained']
    counts.update(retained_claims=len(retained), retained_cases=len({r['case']['case_id'] for r in retained}),
                  review_queue=status['review'], merged_duplicates=status['merged'])
    _log(f"stages 4-5: {counts['candidate_records']} candidate records -> {counts['retained_claims']} retained claims in "
         f"{counts['retained_cases']} cases, {counts['review_queue']} review queue, {counts['merged_duplicates']} merged")
    for key, value in PAPER_FLOW.items():
        if key in counts and counts[key] != value:
            _log(f'warning: {key} = {counts[key]}, paper {value}')
    return results, counts


def prompt_record(result):
    """The source fields one claim's prompts need (contains case text; keep local)."""
    case = result['case']
    return {'unit_id': result['unit_id'], 'case_id': case['case_id'], 'judgment_date': case['judgment_date'],
            'female_party': case['female_party'], 'party_a_name': case['party_a_name'],
            'party_b_name': case['party_b_name'], 'party_a_gender': case['party_a_gender'],
            'party_b_gender': case['party_b_gender'],
            'header_texts': [h['text'] for h in case['header_evidence']],
            **result['fields']}


def _units_frame(results):
    from src.prompts.divorce import build_request
    rows = []
    for r in results:
        if r['status'] != 'retained':
            continue
        case = r['case']
        user = build_request(prompt_record(r), 'C0')['parts'][0]['text']
        case_part = user.split('【本次裁决选项】')[0]
        female = male = 0
        pieces = re.split(r'(【[^】]+】)', case_part)
        for head, body in zip(pieces[1::2], pieces[2::2]):
            if re.match(r'【(原告|被告)（女方）', head):
                female += sum(not c.isspace() for c in body)
            elif re.match(r'【(原告|被告)（男方）', head):
                male += sum(not c.isspace() for c in body)
        rows.append({'unit_id': r['unit_id'], 'cluster_id': case['case_id'], 'weight': 1,
                     'real_underdog': int(r['ground_truth']['court_female_support']),
                     'female_plaintiff': int(case['female_party'] == 'A'),
                     'post_divorce_action': int(case['record_type'] == 'post_divorce_property'),
                     'record_length_chars': sum(not c.isspace() for c in case_part),
                     'female_statement_chars': female, 'male_statement_chars': male})
    return pd.DataFrame(rows, columns=UNIT_COLUMNS).sort_values('unit_id').reset_index(drop=True)


def build_units(raw_dir, cache_dir=None):
    """Units of the full constructed sample, with the columns of ``units_divorce.csv``.

    ``real_underdog`` is 1 when the court's claim-level ruling favors the female spouse.
    Record-length features count non-whitespace characters of the C0 user message before
    the decision options; the spouse statement lengths sum the sections headed with that
    spouse's role.
    """
    results, _ = build_sample(raw_dir, cache_dir)
    return _units_frame(results)


def load_records(raw_dir, unit_ids=None, cache_dir=None):
    """Map unit_id -> source fields for ``src.prompts.divorce.build_request``."""
    results, _ = build_sample(raw_dir, cache_dir)
    wanted = None if unit_ids is None else set(unit_ids)
    out = {r['unit_id']: prompt_record(r) for r in results
           if r['status'] == 'retained' and (wanted is None or r['unit_id'] in wanted)}
    if wanted is not None and set(out) != wanted:
        raise KeyError(f'{len(wanted - set(out))} requested units are not in the rebuilt sample')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Rebuild the divorce-property claim sample.')
    parser.add_argument('raw_dir', help='folder holding the annual cncases ZIP archives')
    parser.add_argument('--cache', default=None, help='folder for the divorce-row cache (default raw_dir/divorce_rows)')
    parser.add_argument('--out', default=None, help='write the units CSV here')
    args = parser.parse_args()
    frame = build_units(args.raw_dir, args.cache)
    if args.out:
        frame.to_csv(args.out, index=False, lineterminator='\n')
    print(f'{len(frame)} claims in {frame.cluster_id.nunique()} cases')
