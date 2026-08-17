import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================================================
# 1. 参数设置
# =========================================================

SEARCH_URL = (
    "https://wzdig.pbc.gov.cn/search/pcRender"
    "?pageId=c177a85bd02b4114bebebd210809f691"
)

SEARCH_WORD = "货币政策委员会召开"

# 最早和最晚年份，可自行调整
MIN_YEAR = 2011
MAX_YEAR = 2026

# 防止无限翻页
MAX_SEARCH_PAGES = 12

# 输出文件夹
OUTPUT_DIR = Path("data/monetary_policy_meetings")
TXT_DIR = OUTPUT_DIR / "raw_txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    "Referer": SEARCH_URL,
}


# =========================================================
# 2. 基础函数
# =========================================================

def clean_text(text):
    """清理空格和多余换行。"""
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def safe_filename(text):
    """删除文件名中不允许出现的字符。"""
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    return text.strip()


def extract_year_quarter(title):
    """从标题中提取年份和季度。"""
    year_match = re.search(r"(19\d{2}|20\d{2})年", title)
    quarter_match = re.search(r"第([一二三四1234])季度", title)

    if not year_match or not quarter_match:
        return None, None

    year = int(year_match.group(1))

    quarter_map = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
    }

    quarter = quarter_map[quarter_match.group(1)]

    return year, quarter


def is_target_title(title):
    """
    判断是否为货币政策委员会季度例会通稿。
    规则故意写得稍宽松，以兼容早期标题格式。
    """
    normalized = re.sub(r"\s+", "", title)

    conditions = [
        "货币政策委员会" in normalized,
        "季度" in normalized,
        "例会" in normalized,
        re.search(r"(19\d{2}|20\d{2})年", normalized) is not None,
    ]

    return all(conditions)


# =========================================================
# 3. 获取搜索表单参数
# =========================================================

def get_search_form_data(session):
    response = session.get(
        SEARCH_URL,
        headers=HEADERS,
        timeout=40,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "lxml")

    # 读取整个页面的隐藏参数，而不只读取form内部
    form_data = {}

    for input_tag in soup.select("input[name]"):
        name = input_tag.get("name")
        value = input_tag.get("value", "")

        if name:
            form_data[name] = value

    if "app" not in form_data:
        raise RuntimeError("未读取到央行搜索页的app参数。")

    return form_data


# =========================================================
# 4. 搜索并提取公告链接
# =========================================================

def search_one_page(
    session,
    base_form_data,
    search_word,
    page_number,
):
    data = base_form_data.copy()

    data.update(
        {
            "q": search_word,
            "searchWord": search_word,
            "originalSearch": search_word,
            "pNo": str(page_number),

            # 标题检索
            "searchArea": "title",

            # 发布时间降序
            "sr": "dateTime desc",

            "advSearch": "",
            "qAnd": "",
            "qOr": "",
            "qAll": "",
            "qNot": "",
        }
    )

    response = session.post(
        SEARCH_URL,
        data=data,
        headers={
            **HEADERS,
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://wzdig.pbc.gov.cn",
        },
        timeout=60,
        allow_redirects=True,
    )

    response.raise_for_status()

    # 央行搜索服务有时会用HTTP 200返回业务参数错误页，
    # 因而不能只依赖raise_for_status()判断请求是否成功。
    if "error parameters!" in response.text:
        raise RuntimeError(
            "央行搜索接口返回参数错误："
            f"搜索词={search_word}，页码={page_number}"
        )

    response.encoding = response.apparent_encoding

    return response.text


def parse_search_results(html):
    soup = BeautifulSoup(html, "lxml")

    results = []
    seen_urls = set()

    # 不再依赖div.searchMod，直接检查所有链接
    for link_tag in soup.find_all("a", href=True):
        title = clean_text(
            link_tag.get_text(" ", strip=True)
        )

        url = urljoin(
            SEARCH_URL,
            link_tag.get("href", "")
        )

        # 只保留央行网站链接
        if "pbc.gov.cn" not in url:
            continue

        # 必须同时包含这些内容
        normalized_title = re.sub(r"\s+", "", title)

        if "货币政策委员会" not in normalized_title:
            continue

        if "季度" not in normalized_title:
            continue

        if "例会" not in normalized_title:
            continue

        year, quarter = extract_year_quarter(
            normalized_title
        )

        if year is None or quarter is None:
            continue

        if not (MIN_YEAR <= year <= MAX_YEAR):
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)

        # 从链接附近寻找发布日期
        parent = link_tag.find_parent(
            ["div", "li", "td"]
        )

        if parent:
            nearby_text = clean_text(
                parent.get_text(" ", strip=True)
            )
        else:
            nearby_text = normalized_title

        date_match = re.search(
            r"(19\d{2}|20\d{2})年"
            r"(\d{1,2})月"
            r"(\d{1,2})日",
            nearby_text,
        )

        publish_date = None

        if date_match:
            publish_date = (
                f"{date_match.group(1)}-"
                f"{int(date_match.group(2)):02d}-"
                f"{int(date_match.group(3)):02d}"
            )

        results.append(
            {
                "year": year,
                "quarter": quarter,
                "publish_date": publish_date,
                "title": title,
                "url": url,
            }
        )

    return results


QUARTER_MAP = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
}


def collect_all_links(session):
    base_form_data = get_search_form_data(session)

    all_results = []
    seen_urls = set()

    for year in range(MAX_YEAR, MIN_YEAR - 1, -1):
        for target_quarter, quarter_cn in QUARTER_MAP.items():
            search_word = (
                f"货币政策委员会 "
                f"{year}年"
                f"第{quarter_cn}季度 "
                f"例会"
            )

            print(
                f"\n正在搜索 {year} 年"
                f"第{target_quarter}季度……"
            )

            quarter_found = 0

            for page_number in range(
                1,
                MAX_SEARCH_PAGES + 1,
            ):
                print(
                    f"读取 {year} 年Q{target_quarter}"
                    f"搜索结果第 {page_number} 页……"
                )

                try:
                    html = search_one_page(
                        session=session,
                        base_form_data=base_form_data,
                        search_word=search_word,
                        page_number=page_number,
                    )

                    # 保存第一个搜索结果页面，方便检查
                    if (
                        year == MAX_YEAR
                        and target_quarter == 1
                        and page_number == 1
                    ):
                        debug_file = (
                            OUTPUT_DIR /
                            "debug_search_result.html"
                        )

                        debug_file.write_text(
                            html,
                            encoding="utf-8",
                        )

                        print(
                            "调试页面已保存："
                            f"{debug_file}"
                        )

                    page_results = parse_search_results(html)

                except Exception as error:
                    print(
                        f"{year}年Q{target_quarter}"
                        f"第{page_number}页失败：{error}"
                    )
                    continue

                for item in page_results:
                    # 只保留当前查询的年份和季度
                    if item["year"] != year:
                        continue

                    if item["quarter"] != target_quarter:
                        continue

                    if item["url"] in seen_urls:
                        continue

                    seen_urls.add(item["url"])
                    all_results.append(item)
                    quarter_found += 1

                    print(
                        f"找到：{item['year']} "
                        f"Q{item['quarter']} "
                        f"{item['title']}"
                    )

                # 当前查询找到候选公告后即可停止翻页。
                # 候选结果会在主程序中按主站优先去重。
                if quarter_found >= 1:
                    break

                time.sleep(1.5)

            print(
                f"{year}年Q{target_quarter}"
                f"共找到 {quarter_found} 条新结果。"
            )

            time.sleep(1.5)

    return all_results

# =========================================================
# 5. 提取单篇公告正文
# =========================================================

def remove_unwanted_elements(soup):
    """删除脚本、菜单和页脚等无关内容。"""
    selectors = [
        "script",
        "style",
        "noscript",
        "iframe",
        "nav",
        "header",
        "footer",
        ".header",
        ".footer",
        ".nav",
        ".menu",
        ".copyright",
        ".print",
        ".share",
    ]

    for selector in selectors:
        for tag in soup.select(selector):
            tag.decompose()


def extract_article_text(html, expected_title):
    """从央行公告页面中提取正文。"""
    soup = BeautifulSoup(html, "lxml")
    remove_unwanted_elements(soup)

    # 人民银行不同时期网页可能使用不同正文选择器
    candidate_selectors = [
        ".TRS_Editor",
        "#zoom",
        "#Zoom",
        ".article-content",
        ".article_content",
        ".content",
        ".content_main",
        ".detail",
        ".detail_content",
        ".news_content",
        ".text",
    ]

    candidates = []

    for selector in candidate_selectors:
        for tag in soup.select(selector):
            paragraphs = [
                clean_text(p.get_text(" ", strip=True))
                for p in tag.select("p")
            ]

            paragraphs = [
                p for p in paragraphs
                if len(p) >= 8
            ]

            if paragraphs:
                text = "\n\n".join(paragraphs)
            else:
                text = clean_text(tag.get_text("\n", strip=True))

            if len(text) >= 100:
                candidates.append(text)

    # 如果指定选择器没有找到正文，则从所有段落中提取
    if not candidates:
        paragraphs = []

        for p in soup.find_all("p"):
            text = clean_text(p.get_text(" ", strip=True))

            if len(text) < 8:
                continue

            if any(
                unwanted in text
                for unwanted in [
                    "设为首页",
                    "加入收藏",
                    "法律声明",
                    "版权所有",
                    "打印本页",
                    "关闭窗口",
                    "网站地图",
                    "京ICP备",
                ]
            ):
                continue

            paragraphs.append(text)

        if paragraphs:
            candidates.append("\n\n".join(paragraphs))

    if not candidates:
        return ""

    # 一般正文是候选区域中长度最长的一块
    article_text = max(candidates, key=len)

    # 删除可能重复出现的标题
    article_text = article_text.replace(expected_title, "", 1)

    return clean_text(article_text)


def download_article(session, item):
    """下载一篇公告并提取正文。"""
    response = session.get(
        item["url"],
        headers=HEADERS,
        timeout=40,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding

    text = extract_article_text(
        html=response.text,
        expected_title=item["title"],
    )

    if len(text) < 100:
        raise ValueError(
            f"提取出的正文过短，只有 {len(text)} 个字符"
        )

    return text


# =========================================================
# 6. 保存结果
# =========================================================

def save_txt(item):
    """将一篇公告保存成TXT。"""
    filename = (
        f"{item['year']}_Q{item['quarter']}_"
        f"{safe_filename(item['title'])}.txt"
    )

    filepath = TXT_DIR / filename

    content = (
        f"标题：{item['title']}\n"
        f"年份：{item['year']}\n"
        f"季度：Q{item['quarter']}\n"
        f"发布日期：{item.get('publish_date') or ''}\n"
        f"原始网址：{item['url']}\n\n"
        f"{item['text']}"
    )

    filepath.write_text(content, encoding="utf-8")

    return str(filepath)


# =========================================================
# 7. 主程序
# =========================================================

def main():
    session = requests.Session()

    print("第一步：搜索季度例会通稿")
    links = collect_all_links(session)

    if not links:
        print("没有找到符合条件的公告。")
        print("请检查网络连接或央行搜索页面是否发生变化。")
        return

    # 先根据年份、季度和URL去重
    link_df = pd.DataFrame(links)

    # 同一季度可能存在改版前后的重复URL或转载页面。
    # 央行总行主站优先，然后每个年份、季度只保留一个结果。
    link_df["is_main_site"] = (
        link_df["url"]
        .str.startswith("https://www.pbc.gov.cn/")
        .astype(int)
    )

    link_df = link_df.sort_values(
        ["year", "quarter", "is_main_site"],
        ascending=[True, True, False],
    )

    link_df = link_df.drop_duplicates(
        subset=["year", "quarter"],
        keep="first",
    )

    link_df = link_df.sort_values(
        ["year", "quarter"]
    )

    link_file = OUTPUT_DIR / "meeting_links.csv"
    link_df.to_csv(
        link_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n共找到 {len(link_df)} 条候选链接。")
    print(f"链接表已保存到：{link_file}")

    print("\n第二步：逐篇下载公告正文")

    completed = []
    failed = []

    for index, row in link_df.iterrows():
        item = row.to_dict()

        print(
            f"正在下载：{item['year']} Q{item['quarter']} "
            f"{item['title']}"
        )

        try:
            text = download_article(session, item)
            item["text"] = text
            item["text_length"] = len(text)
            item["txt_path"] = save_txt(item)

            completed.append(item)

            print(f"成功，正文长度：{len(text)}")

        except Exception as error:
            failed.append(
                {
                    **item,
                    "error": str(error),
                }
            )

            print(f"失败：{error}")

        time.sleep(1.5)

    # 保存成功结果
    if completed:
        completed_df = pd.DataFrame(completed)

        completed_df = completed_df.sort_values(
            ["year", "quarter"]
        )

        completed_file = (
            OUTPUT_DIR / "monetary_policy_meetings.csv"
        )

        completed_df.to_csv(
            completed_file,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\n成功下载 {len(completed_df)} 篇，"
            f"汇总文件：{completed_file}"
        )

    # 保存失败结果
    if failed:
        failed_df = pd.DataFrame(failed)

        failed_file = OUTPUT_DIR / "failed_urls.csv"

        failed_df.to_csv(
            failed_file,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"有 {len(failed_df)} 篇失败，"
            f"失败记录：{failed_file}"
        )

    print("\n全部运行结束。")


if __name__ == "__main__":
    main()
