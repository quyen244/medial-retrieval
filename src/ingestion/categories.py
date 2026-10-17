"""Which language each source domain publishes in.

The retrieval task is cross-lingual: one query may have relevant answers in
Vietnamese, Chinese and English at once. Knowing a document's language is
therefore part of its identity, not an afterthought, and it has to be decided
by something more reliable than the page's own `<html lang>` attribute, which
is frequently missing or simply wrong.

The domain is that reliable signal. Every domain in this corpus publishes in
exactly one language, so the mapping is fixed and auditable here rather than
guessed per page.
"""

VIETNAMESE = "vi"
CHINESE = "zh"
ENGLISH = "en"

# Everything not listed below is Vietnamese; those are the majority and they
# are overwhelmingly .vn domains.
CHINESE_DOMAINS = frozenset({
    "www.120ask.com",
    "www.cnkang.com",
    "ask.39.net",
    "www.familydoctor.com.cn",
    "baidianfeng.familydoctor.com.cn",
    "qy.familydoctor.com.cn",
    "ask.familydoctor.com.cn",
    "www.familydoctor.cn",
    "zysjonline.com",
    "zhongyibaodian.net",
    "www.a-hospital.com",
    "www.youlai.cn",
    "www.msdmanuals.cn",
    "www.wujue.com",
    "www.zydcd.com",
    "qihuangzhishu.com",
    "health.people.com.cn",
    "www.baidu.com",
    "article.iiyi.com",
    "bingli.iiyi.com",
    "test.pmphai.com",
    "jb39.com",
    # 39.net runs one subdomain per specialty.
    "baby.39.net", "jbk.39.net", "care.39.net", "heart.39.net", "fk.39.net",
    "nk.39.net", "cancer.39.net", "tnb.39.net", "woman.39.net", "food.39.net",
    "ek.39.net", "man.39.net", "shen.39.net", "gan.39.net", "wei.39.net",
    "sj.39.net", "fitness.39.net", "gk.39.net", "pf.39.net", "gc.39.net",
    "byby.39.net",
})

ENGLISH_DOMAINS = frozenset({
    "pubmed.ncbi.nlm.nih.gov",
})


def language_of(domain: str) -> str:
    """The language a domain publishes in: 'vi', 'zh' or 'en'."""
    domain = domain.lower()
    if domain in ENGLISH_DOMAINS:
        return ENGLISH
    if domain in CHINESE_DOMAINS:
        return CHINESE
    return VIETNAMESE
