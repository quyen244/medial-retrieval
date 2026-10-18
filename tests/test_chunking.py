import unittest
from src.rag.chunker import (
    clean_markdown_syntax,
    VietnameseMarkdownCleaner,
    VietnameseHeadingPromoter,
    VietnameseSentenceSplitter,
    VietnameseHierarchicalChunker,
    ChineseMarkdownCleaner,
    ChineseHeadingPromoter,
    ChineseSentenceSplitter,
    ChineseHierarchicalChunker,
    EnglishMarkdownCleaner,
    EnglishHeadingPromoter,
    EnglishSentenceSplitter,
    EnglishHierarchicalChunker,
    ChunkerRegistry,
    ChunkerDispatcher,
    BaseChunker,
    Chunk,
)
from src.rag.chunker.chinese.cleaner import clean_markdown_syntax as clean_markdown_syntax_zh
from src.rag.chunker.english.cleaner import clean_markdown_syntax as clean_markdown_syntax_en



class TestVietnameseHeadingPromoter(unittest.TestCase):
    def setUp(self):
        self.promoter = VietnameseHeadingPromoter()

    def test_numeric_promotion(self):
        text = "Mở đầu bài viết.\n\n**1. Không ăn mỡ động vật**\n\nĐây là nội dung giải thích."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("## 1. Không ăn mỡ động vật", res)
        self.assertNotIn("**1. Không ăn mỡ động vật**", res)

    def test_roman_promotion(self):
        text = "Mở đầu.\n\n**IV) Khuyến nghị của chuyên gia y tế**\n\nNội dung mục."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("## IV) Khuyến nghị của chuyên gia y tế", res)

    def test_false_positive_filter_intermediate_period(self):
        text = "Mở đầu.\n\n**1. Bác sĩ A. Đã chia sẻ kinh nghiệm**\n\nNội dung."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)
        self.assertIn("**1. Bác sĩ A. Đã chia sẻ kinh nghiệm**", res)

    def test_false_positive_filter_length(self):
        long_line = "**1. " + "A" * 120 + "**"
        text = f"Mở đầu.\n\n{long_line}\n\nNội dung."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_false_positive_inline_bold(self):
        text = "Trong bữa ăn, bạn nên nhớ **1. Không ăn mỡ động vật** để giữ sức khỏe."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_zero_headings_safe(self):
        text = "Văn bản hoàn toàn bình thường không có bất kỳ heading nào."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)
        self.assertEqual(stats["headings_after"], 0)
        self.assertEqual(res, text)


class TestVietnameseSentenceSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = VietnameseSentenceSplitter()

    def test_abbreviation_protection(self):
        text = "PGS.TS. Vũ Văn Giáp chia sẻ kinh nghiệm tại hội thảo. ThS.BS. Nguyễn Văn A cũng tham dự."
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertIn("PGS.TS. Vũ Văn Giáp", sents[0])
        self.assertIn("ThS.BS. Nguyễn Văn A", sents[1])

    def test_decimal_and_ellipsis_protection(self):
        text = "Bệnh viện đạt 99.5% tiêu chuẩn... Giá trị gói xét nghiệm là 1.500.000 đồng! Bạn thấy thế nào?"
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 3)
        self.assertEqual(sents[0], "Bệnh viện đạt 99.5% tiêu chuẩn...")
        self.assertEqual(sents[1], "Giá trị gói xét nghiệm là 1.500.000 đồng!")
        self.assertEqual(sents[2], "Bạn thấy thế nào?")


class TestVietnameseHierarchicalChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = VietnameseHierarchicalChunker(
            min_chunk_size=100,
            target_chunk_size=200,
            max_chunk_size=350,
            overlap_ratio=0.15,
        )

    def test_chunking_with_promoted_heading(self):
        doc = {
            "source": {
                "doc_id": 999,
                "domain": "test.vn",
                "final_url": "https://test.vn/article",
                "title": "Bài test",
                "language": "vi",
            },
            "content": {
                "markdown": "# Tiêu đề chính\n\nĐoạn mở đầu của bài viết.\n\n**1. Chế độ ăn uống**\n\nĂn nhiều rau xanh và uống đủ nước mỗi ngày. Giảm bớt lượng đường và muối tinh chế.\n\n**2. Luyện tập thể thao**\n\nĐi bộ ít nhất 30 phút mỗi ngày. Chạy bộ hoặc tập yoga đều rất tốt cho tim mạch."
            }
        }
        chunks = self.chunker.chunk(doc)
        self.assertGreater(len(chunks), 0)
        # Verify metadata
        for c in chunks:
            self.assertEqual(c.doc_id, 999)
            self.assertEqual(c.domain, "test.vn")
            self.assertTrue(c.chunk_id.startswith("999_c"))
            self.assertGreater(c.char_length, 0)

    def test_zero_heading_document(self):
        doc = {
            "source": {"doc_id": 100, "domain": "test.vn", "title": "Không heading"},
            "content": {
                "markdown": "Đoạn văn ngắn duy nhất không chứa bất kỳ heading hay ký tự đặc biệt nào."
            }
        }
        chunks = self.chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)
    def test_oversized_unbroken_text_sliced_safely(self):
        doc = {
            "source": {"doc_id": 101, "domain": "test.vn", "title": "Blob rác"},
            "content": {
                # 3000 continuous characters without spaces or punctuation
                "markdown": "A" * 3000
            }
        }
        chunks = self.chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_length, self.chunker.max_chunk_size)


class TestOCPDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = ChunkerDispatcher(default_language="vi")

    def test_locale_tag_normalization(self):
        doc_vi = {"source": {"doc_id": 1, "language": "vi-VN"}, "content": {"markdown": "Tiếng Việt"}}
        doc_en = {"source": {"doc_id": 2, "language": "en-US"}, "content": {"markdown": "English text"}}
        self.assertEqual(self.dispatcher.detect_language(doc_vi), "vi")
        self.assertEqual(self.dispatcher.detect_language(doc_en), "en")

    def test_language_category_takes_priority_over_noisy_language(self):
        """Regression test: source.language_category is curated ground truth and must win
        over the frequently-wrong source.language (e.g. ask.39.net/www.youlai.cn are Chinese
        content mislabeled language='en'; qdnd.vn/benhvienvietduc.org are Vietnamese content
        mislabeled language='en-US')."""
        doc_zh_mislabeled = {"source": {"language": "en", "language_category": "zh"}}
        doc_vi_mislabeled = {"source": {"language": "en-US", "language_category": "vi"}}
        self.assertEqual(self.dispatcher.detect_language(doc_zh_mislabeled), "zh")
        self.assertEqual(self.dispatcher.detect_language(doc_vi_mislabeled), "vi")

    def test_dispatch_vi(self):
        doc = {
            "source": {"doc_id": 1, "language": "vi"},
            "content": {"markdown": "# Tiêu đề tiếng Việt\n\nNội dung ngắn."}
        }
        chunks = self.dispatcher.dispatch(doc)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].language, "vi")

    def test_dispatch_en(self):
        doc = {
            "source": {"doc_id": 2, "language": "en"},
            "content": {"markdown": "# English Title\n\nThis is an English paragraph about health."}
        }
        chunks = self.dispatcher.dispatch(doc)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].language, "en")

    def test_dispatch_zh(self):
        doc = {
            "source": {"doc_id": 3, "language": "zh"},
            "content": {"markdown": "# 中文标题\n\n这是一个关于健康生活的中文段落。保持良好的生活习惯非常重要。"}
        }
        chunks = self.dispatcher.dispatch(doc)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].language, "zh")

    def test_ocp_dynamic_registration_without_modifying_dispatcher(self):
        """Verify Open/Closed Principle: A new language (e.g. 'ja') can be plugged in dynamically."""
        @ChunkerDispatcher.register("ja")
        class JapaneseChunker(BaseChunker):
            def chunk(self, doc, **kwargs):
                return [Chunk(chunk_id="ja_01", chunk_index=0, content="こんにちは", char_length=5, doc_id=doc.get("source", {}).get("doc_id"), language="ja")]
            def chunk_text(self, text, metadata=None, **kwargs):
                return []
            def get_params(self):
                return {}

        doc = {"source": {"doc_id": 99, "language": "ja"}, "content": {"markdown": "テスト"}}
        chunks = self.dispatcher.dispatch(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].language, "ja")
        self.assertEqual(chunks[0].chunk_id, "ja_01")


class TestVietnameseMarkdownCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = VietnameseMarkdownCleaner()

    def test_clean_markdown_syntax_headings_and_bold(self):
        text = "## **1. Húng quế (Ocimum basilicum-Lamiaceae)**"
        cleaned = clean_markdown_syntax(text)
        self.assertEqual(cleaned, "1. Húng quế (Ocimum basilicum-Lamiaceae)")
        self.assertNotIn("**", cleaned)
        self.assertNotIn("##", cleaned)

    def test_clean_markdown_syntax_inline_formatting(self):
        text = "Tinh dầu **húng quế** rất tốt cho *sức khỏe*. __Lưu ý:__ Uống đủ nước."
        cleaned = clean_markdown_syntax(text)
        self.assertEqual(cleaned, "Tinh dầu húng quế rất tốt cho sức khỏe. Lưu ý: Uống đủ nước.")

    def test_clean_markdown_syntax_preserves_numbers_and_emails(self):
        text = "Đạt 99.5%, giá 1.500.000 VNĐ, liên hệ bs_lan@bv.vn hoặc phòng 10-A."
        cleaned = clean_markdown_syntax(text)
        self.assertEqual(cleaned, text)


class TestHeadingToContentBinding(unittest.TestCase):
    def setUp(self):
        self.chunker = VietnameseHierarchicalChunker(
            min_chunk_size=150,
            target_chunk_size=250,
            max_chunk_size=350,
            overlap_ratio=0.10,
        )

    def test_heading_never_orphaned_at_chunk_end(self):
        doc = {
            "source": {"doc_id": 888, "title": "Test Binding"},
            "content": {
                "markdown": (
                    "# Tiêu đề\n\n"
                    "Đoạn văn mở đầu có độ dài vừa đủ để lấp đầy khoảng 150 ký tự trong bộ đệm hiện tại.\n\n"
                    "## **Mục 1: Tiêu đề quan trọng**\n\n"
                    "Nội dung giải thích chi tiết cho Mục 1 đi kèm ngay sau đó."
                )
            }
        }
        chunks = self.chunker.chunk(doc)
        for c in chunks:
            lines = [l.strip() for l in c.content.split("\n") if l.strip()]
            # Ensure the last line of any chunk is not an orphan heading
            last_line = lines[-1]
            self.assertNotEqual(last_line, "Mục 1: Tiêu đề quan trọng")
            self.assertFalse(last_line.startswith("##"))


class TestCrossSectionGreedyMerge(unittest.TestCase):
    def setUp(self):
        self.chunker = VietnameseHierarchicalChunker(
            min_chunk_size=400,
            target_chunk_size=600,
            max_chunk_size=900,
            overlap_ratio=0.10,
        )

    def test_small_sections_merged_into_target_size(self):
        # 5 small sections of ~80 characters each
        sections = [
            f"## Mục {i}\n\nNội dung chi tiết ngắn gọn của mục số {i} nhằm kiểm tra cơ chế gộp greedy."
            for i in range(1, 6)
        ]
        doc = {
            "source": {"doc_id": 777, "title": "Test Merge"},
            "content": {"markdown": "\n\n".join(sections)}
        }
        chunks = self.chunker.chunk(doc)
        # Instead of 5 separate chunks, greedy merge should combine them into fewer, target-sized chunks
        self.assertLess(len(chunks), 4)
        for c in chunks:
            self.assertLessEqual(c.char_length, self.chunker.max_chunk_size)


class TestChineseMarkdownCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = ChineseMarkdownCleaner()

    def test_clean_markdown_syntax_headings_and_bold(self):
        text = "## **一、目的：**"
        cleaned = clean_markdown_syntax_zh(text)
        self.assertEqual(cleaned, "一、目的：")
        self.assertNotIn("**", cleaned)
        self.assertNotIn("##", cleaned)

    def test_clean_markdown_syntax_inline_formatting_no_word_boundary_issue(self):
        # CJK text has no whitespace around emphasis markers; the cleaner must still strip them.
        text = "本病情况**单纯**依靠中医治疗效果较差。"
        cleaned = clean_markdown_syntax_zh(text)
        self.assertEqual(cleaned, "本病情况单纯依靠中医治疗效果较差。")
        self.assertNotIn("*", cleaned)

    def test_clean_markdown_syntax_strips_footnote_markup(self):
        text = "绒毛膜癌并迅速扩散<sup>\\[1\\]</sup><sup>\\[2\\]</sup>。"
        cleaned = clean_markdown_syntax_zh(text)
        self.assertNotIn("<sup>", cleaned)
        self.assertNotIn("</sup>", cleaned)
        self.assertIn("绒毛膜癌并迅速扩散", cleaned)

    def test_clean_markdown_syntax_strips_backlink_arrow(self):
        text = "1. ↑ Seckl MJ, Sebire NJ. Lancet. 2010."
        cleaned = clean_markdown_syntax_zh(text)
        self.assertNotIn("↑", cleaned)

    def test_clean_markdown_syntax_preserves_decimals(self):
        text = "血糖为99.5%，剂量为1.5MU/d。"
        cleaned = clean_markdown_syntax_zh(text)
        self.assertEqual(cleaned, text)

    def test_unescapes_backslash_escaped_tilde(self):
        """Regression for www.youlai.cn 3984400: the HTML->Markdown converter backslash-escapes
        punctuation like '~' to prevent it being misread as strikethrough syntax. That escape is
        a conversion artifact never present in the rendered page and must not leak into chunks."""
        text = "有时10\\~几十不等，而感冒一般7\\~10天可自己恢复。"
        cleaned = clean_markdown_syntax_zh(text)
        self.assertNotIn("\\~", cleaned)
        self.assertIn("10~几十不等", cleaned)
        self.assertIn("7~10天", cleaned)

    def test_unescapes_other_commonmark_escapes_without_misfiring(self):
        """Generalizes beyond '~': other escaped punctuation observed in the zh corpus (ordered-
        list periods, brackets, underscores, angle brackets) must also be un-escaped cleanly."""
        self.assertEqual(clean_markdown_syntax_zh("1\\. 定期进行妇科检查"), "1. 定期进行妇科检查")
        self.assertEqual(clean_markdown_syntax_zh("鼻\\[窦炎、扁桃体炎"), "鼻[窦炎、扁桃体炎")
        self.assertEqual(clean_markdown_syntax_zh("休息\\_单纯隆鼻术者"), "休息_单纯隆鼻术者")
        self.assertEqual(clean_markdown_syntax_zh("血糖浓度\\<2.5mmol/L"), "血糖浓度<2.5mmol/L")

    def test_unescape_ordering_preserves_literal_escaped_emphasis_markers(self):
        """Ordering regression: un-escaping must not run before emphasis-stripping. An
        intentionally-escaped literal '\\*text\\*' (meant as literal asterisks, not markdown
        italics) would otherwise be exposed to the emphasis regexes as bare '*text*' and get
        stripped down to plain 'text', destroying the literal asterisks."""
        self.assertEqual(clean_markdown_syntax_zh("\\*text\\*"), "*text*")
        # A literal escaped '*' coexisting on the same line as real **bold** markup must not let
        # the bold-stripping regex misfire across the escaped markers.
        text = "公式为 a\\*b\\*c，并且**真正加粗**的内容"
        cleaned = clean_markdown_syntax_zh(text)
        self.assertEqual(cleaned, "公式为 a*b*c，并且真正加粗的内容")


class TestChineseHeadingPromoter(unittest.TestCase):
    def setUp(self):
        self.promoter = ChineseHeadingPromoter()

    def test_plain_chinese_numeral_promotion(self):
        text = "上文内容。\n\n一、辨证选方\n\n治法：温肾扶阳，活血行瘀。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("一、辨证选方", res)
        self.assertIn("####", res)

    def test_parenthesized_chinese_numeral_promotion(self):
        text = "上文内容。\n\n(一)发病原因\n\n发生原因至今不明。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("(一)发病原因", res)

    def test_circled_numeral_promotion(self):
        text = "上文。\n\n①术前准备：\n\n详细了解患者一般情况。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)

    def test_bold_wrapped_promotion(self):
        text = "上文。\n\n**一、目的：**\n\n对症治疗以减轻痛苦。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("一、目的：", res)
        self.assertNotIn("**一、目的：**", res)

    def test_false_positive_fused_list_item(self):
        """A numbered list item whose body is fused on the same line/paragraph (not a
        standalone short heading) must not be promoted."""
        text = (
            "上文。\n\n"
            "1.营养不良学说 实验动物中缺乏叶酸可致胚胎死亡，推测母体缺乏叶酸可能和滋养细胞肿瘤的发生有关。\n\n"
            "下文。"
        )
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_false_positive_punctuation_in_body(self):
        text = "上文。\n\n（2）连续应用可有嗜睡、乏力、疲劳，偶有腹泻；\n\n下文。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_false_positive_coordinate_list_item_ending_in_semicolon(self):
        """A list item with no comma/period but ending in a coordinating semicolon (signalling
        'more parallel items follow') must not be promoted - it is one item of a list, not a
        heading for one. Regression for a real false positive found in www.a-hospital.com."""
        text = "上文。\n\n（3）使用IFNα期间不宜驾驶车辆及操作机器；\n\n下文。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_false_positive_inline_bold_fused(self):
        text = "上文。\n\n**一、  和年龄有关。**在屈光发育稳定期之前，远视度数会自行降低。\n\n下文。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_zero_headings_safe(self):
        text = "普通的中文段落，不包含任何编号或标题。"
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)
        self.assertEqual(res, text)


class TestChineseSentenceSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = ChineseSentenceSplitter()

    def test_decimal_and_latin_abbreviation_protection(self):
        text = "患者血糖为99.5%，病情稳定。IFNα-2a使用剂量为1.5MU/d。你好吗？是的！"
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 4)
        self.assertEqual(sents[0], "患者血糖为99.5%，病情稳定。")
        self.assertEqual(sents[1], "IFNα-2a使用剂量为1.5MU/d。")
        self.assertEqual(sents[2], "你好吗？")
        self.assertEqual(sents[3], "是的！")

    def test_ellipsis_normalization(self):
        text = "这是一个例子......后续内容。"
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertTrue(sents[0].startswith("这是一个例子"))

    def test_preserves_paragraph_boundary_newline_between_sentences(self):
        """Regression for qihuangzhishu.com 877312: a lone '\\n' between sentences marks a
        blank-line-separated paragraph/list-item boundary collapsed earlier in the pipeline (by
        MarkdownHeaderTextSplitter) and must survive, or "".join(sentences) at chunker call sites
        fuses unrelated items together (the circled numerals ①②③ ran together with zero
        separator before this fix)."""
        text = (
            "①必需适合身体需要的平衡膳食，含有充足的各种营养素。\n"
            "②一般正常的食品均可采用。\n"
            "③避免应用强烈辛辣刺激性的食品或调味品。"
        )
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 3)
        self.assertTrue(sents[0].endswith("\n"))
        self.assertTrue(sents[1].endswith("\n"))
        self.assertFalse(sents[2].endswith("\n"))
        # "".join() is the deliberate zero-separator join used throughout chunker.py for normal
        # Chinese prose; it must reproduce the original text exactly once sentences carry the
        # boundary newline themselves.
        self.assertEqual("".join(sents), text)

    def test_no_spurious_newline_for_plain_inline_spacing(self):
        """A sentence boundary with no original newline (ordinary intra-paragraph Chinese prose)
        must stay joined with zero separator - Chinese has no inter-sentence whitespace, so the
        fix must not inject a '\\n' where none existed in the source."""
        text = "患者血糖为99.5%，病情稳定。你好吗？是的！"
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 3)
        for s in sents:
            self.assertFalse(s.endswith("\n"))
        self.assertEqual("".join(sents), text)


class TestChineseHierarchicalChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = ChineseHierarchicalChunker(
            min_chunk_size=100,
            target_chunk_size=200,
            max_chunk_size=350,
            overlap_ratio=0.15,
        )

    def test_chunking_with_promoted_heading(self):
        doc = {
            "source": {
                "doc_id": 998,
                "domain": "test.cn",
                "final_url": "https://test.cn/article",
                "title": "测试文章",
                "language_category": "zh",
            },
            "content": {
                "markdown": (
                    "# 测试文章\n\n开篇段落介绍疾病的基本情况。\n\n"
                    "一、辨证选方\n\n治法：温肾扶阳，活血行瘀，方药为少腹逐瘀汤加减。\n\n"
                    "二、其它疗法\n\n中药大黄蛰虫丸具有祛瘀生新的作用，可用于治疗。"
                )
            }
        }
        chunks = self.chunker.chunk(doc)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c.doc_id, 998)
            self.assertEqual(c.domain, "test.cn")
            self.assertEqual(c.language, "zh")
            self.assertTrue(c.chunk_id.startswith("998_c"))
            self.assertGreater(c.char_length, 0)

    def test_zero_heading_document(self):
        doc = {
            "source": {"doc_id": 200, "domain": "test.cn", "title": "无标题"},
            "content": {"markdown": "这是一段不包含任何标题或特殊字符的普通中文段落内容。"}
        }
        chunks = self.chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_oversized_unbroken_text_sliced_by_character_count(self):
        """CJK has no whitespace word boundary, so the oversized fallback must slice by raw
        character count (not str.split()/word-based slicing like the Vietnamese chunker)."""
        doc = {
            "source": {"doc_id": 201, "domain": "test.cn", "title": "无标点文本块"},
            "content": {"markdown": "甲" * 3000}
        }
        chunks = self.chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_length, self.chunker.max_chunk_size)

    def test_heading_never_orphaned_at_chunk_end(self):
        doc = {
            "source": {"doc_id": 202, "title": "绑定测试"},
            "content": {
                "markdown": (
                    "# 标题\n\n"
                    "开篇段落的长度足够填满当前缓冲区的大部分空间以便测试绑定机制是否正常工作。\n\n"
                    "**一、重要事项：**\n\n"
                    "紧随其后的详细说明内容立刻填补了这个标题。"
                )
            }
        }
        chunks = self.chunker.chunk(doc)
        for c in chunks:
            lines = [l.strip() for l in c.content.split("\n") if l.strip()]
            last_line = lines[-1]
            self.assertNotEqual(last_line, "一、重要事项：")
            self.assertFalse(last_line.startswith("####"))

    def test_sentence_overlap_copy_matches_original_newline_exactly(self):
        """Regression for www.a-hospital.com 2989615: the sentence-level overlap carried into the
        next chunk must be byte-identical to the span it was copied from, including any
        paragraph-boundary '\\n' between its sentences. Before the sentence-splitter fix, the
        overlap copy (built via split_sentences()+"".join()) silently dropped a '\\n' that the
        original chunk text still had, so the same source span disagreed between the two chunks."""
        overlap_chunker = ChineseHierarchicalChunker(
            min_chunk_size=100, target_chunk_size=200, max_chunk_size=350, overlap_ratio=0.9,
        )
        chunk_text = (
            "前文内容介绍疾病的基本情况。\n\n"
            "①原发性血小板增多症：亦首选IFNα，一般治疗3～12个月，血小板可降至正常；亦可与化疗联合应用。\n"
            "（4）其它肿瘤：如黑色素瘤、肾细胞癌等。"
        )
        last_para = chunk_text.split("\n\n")[-1]
        overlap = overlap_chunker._extract_sentence_overlap(chunk_text)
        self.assertIn("\n", overlap)
        self.assertEqual(overlap, last_para)

    def test_max_chunk_size_invariant_holds_with_collapsed_paragraph_newlines(self):
        """The sentence-splitter fix makes a sentence that ends on a collapsed paragraph boundary
        one character longer (it now carries a trailing '\\n'). Confirm this extra character
        never pushes a chunk over max_chunk_size when many such sentences are packed together."""
        item = "①条款内容描述文字共二十个字。"
        body = "\n".join([item] * 40)
        doc = {
            "source": {"doc_id": 203, "title": "条款列表"},
            "content": {"markdown": "# 条款列表\n\n" + body},
        }
        chunks = self.chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_length, self.chunker.max_chunk_size)


class TestEnglishMarkdownCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = EnglishMarkdownCleaner()

    def test_clean_markdown_syntax_headings_and_bold(self):
        text = "## **1. Background of the Study**"
        cleaned = clean_markdown_syntax_en(text)
        self.assertEqual(cleaned, "1. Background of the Study")
        self.assertNotIn("**", cleaned)
        self.assertNotIn("##", cleaned)

    def test_clean_markdown_syntax_inline_formatting(self):
        text = "The **serum level** was *significantly* elevated. __Note:__ Patient improved."
        cleaned = clean_markdown_syntax_en(text)
        self.assertEqual(cleaned, "The serum level was significantly elevated. Note: Patient improved.")

    def test_clean_markdown_syntax_preserves_numbers_and_units(self):
        text = "Ferritin was 296.3 +/- 61.9 ng/mL, p < 0.05, contact dr.smith@hospital.org."
        cleaned = clean_markdown_syntax_en(text)
        self.assertEqual(cleaned, text)


class TestEnglishHeadingPromoter(unittest.TestCase):
    def setUp(self):
        self.promoter = EnglishHeadingPromoter()

    def test_numeric_promotion(self):
        text = "Introductory text.\n\n**1. Study Design**\n\nThis section describes the design."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("## 1. Study Design", res)
        self.assertNotIn("**1. Study Design**", res)

    def test_roman_promotion(self):
        text = "Introductory text.\n\n**IV) Clinical Recommendations**\n\nDetailed content follows."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("## IV) Clinical Recommendations", res)

    def test_letter_promotion(self):
        text = "Introductory text.\n\n**A. Patient Selection**\n\nCriteria are described below."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 1)
        self.assertIn("## A. Patient Selection", res)

    def test_false_positive_filter_intermediate_period(self):
        text = "Intro.\n\n**1. Dr. A. reported findings**\n\nContent."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)
        self.assertIn("**1. Dr. A. reported findings**", res)

    def test_false_positive_filter_length(self):
        long_line = "**1. " + "A" * 120 + "**"
        text = f"Intro.\n\n{long_line}\n\nContent."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_false_positive_inline_bold(self):
        text = "In the study, remember **1. Reduce caloric intake** for best results."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)

    def test_zero_headings_safe(self):
        text = "A perfectly normal paragraph without any heading markers at all."
        res, stats = self.promoter.process(text)
        self.assertEqual(stats["promoted_count"], 0)
        self.assertEqual(res, text)


class TestEnglishSentenceSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = EnglishSentenceSplitter()

    def test_scientific_abbreviation_protection(self):
        text = "Dr. J. Smith et al. reported approx. 15% improvement vs. placebo. This was significant."
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertIn("Dr. J. Smith et al.", sents[0])
        self.assertIn("approx. 15% improvement vs. placebo.", sents[0])

    def test_decimal_and_unit_protection(self):
        text = "Serum ferritin was 296.3 +/- 61.9 ng/mL, p < 0.05. The difference was significant."
        sents = self.splitter.split_sentences(text)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0], "Serum ferritin was 296.3 +/- 61.9 ng/mL, p < 0.05.")
        self.assertEqual(sents[1], "The difference was significant.")


class TestEnglishHierarchicalChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = EnglishHierarchicalChunker(
            min_chunk_size=100,
            target_chunk_size=200,
            max_chunk_size=350,
            overlap_ratio=0.15,
        )

    def test_chunking_with_pubmed_style_structure(self):
        doc = {
            "source": {
                "doc_id": 997,
                "domain": "pubmed.ncbi.nlm.nih.gov",
                "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                "title": "A Study of Disease X",
                "language_category": "en",
            },
            "content": {
                "markdown": (
                    "# A Study of Disease X\n\n## Abstract\n\n"
                    "### Background\n\nDisease X is a rare condition affecting the pulmonary system.\n\n"
                    "### Methods\n\nWe enrolled 50 patients in a prospective cohort study design.\n\n"
                    "### Results\n\nTreatment improved outcomes significantly across all cohorts.\n\n"
                    "## Citation\n\nJ Doe\nJournal of Medicine. 2023.\nPMID: 99999"
                )
            }
        }
        chunks = self.chunker.chunk(doc)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertEqual(c.doc_id, 997)
            self.assertEqual(c.language, "en")
            self.assertTrue(c.chunk_id.startswith("997_c"))
            self.assertGreater(c.char_length, 0)

    def test_zero_heading_document(self):
        doc = {
            "source": {"doc_id": 300, "domain": "test.en", "title": "No heading"},
            "content": {"markdown": "A short single paragraph without any heading or special characters."}
        }
        chunks = self.chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_index, 0)

    def test_oversized_unbroken_text_sliced_safely(self):
        doc = {
            "source": {"doc_id": 301, "domain": "test.en", "title": "Blob"},
            "content": {"markdown": "A" * 3000}
        }
        chunks = self.chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_length, self.chunker.max_chunk_size)

    def test_heading_never_orphaned_at_chunk_end(self):
        doc = {
            "source": {"doc_id": 302, "title": "Binding Test"},
            "content": {
                "markdown": (
                    "# Title\n\n"
                    "An opening paragraph long enough to fill most of the current buffer for this test.\n\n"
                    "## **Section 1: Important Findings**\n\n"
                    "Detailed explanation content for section one follows immediately after."
                )
            }
        }
        chunks = self.chunker.chunk(doc)
        for c in chunks:
            lines = [l.strip() for l in c.content.split("\n") if l.strip()]
            last_line = lines[-1]
            self.assertNotEqual(last_line, "Section 1: Important Findings")
            self.assertFalse(last_line.startswith("##"))


class TestChunkDualAPI(unittest.TestCase):
    def test_object_and_dict_access(self):
        chunk = Chunk(
            chunk_id="test_c001",
            chunk_index=0,
            content="Nội dung kiểm thử",
            char_length=17,
            doc_id=123,
            domain="test.vn",
            title="Tiêu đề",
            metadata={"word_count": 4, "custom_key": "custom_val"}
        )
        # Object attribute access
        self.assertEqual(chunk.content, "Nội dung kiểm thử")
        self.assertEqual(chunk.metadata["word_count"], 4)
        self.assertEqual(chunk.doc_id, 123)

        # Dict / indexing access
        self.assertEqual(chunk["content"], "Nội dung kiểm thử")
        self.assertEqual(chunk["doc_id"], 123)
        self.assertEqual(chunk["domain"], "test.vn")
        self.assertEqual(chunk["word_count"], 4)
        self.assertEqual(chunk["custom_key"], "custom_val")
        self.assertEqual(chunk.get("title"), "Tiêu đề")
        self.assertEqual(chunk.get("nonexistent", "fallback"), "fallback")
        self.assertIn("content", chunk)
        self.assertIn("word_count", chunk)

        # dict(chunk) conversion
        as_dict = dict(chunk)
        self.assertIsInstance(as_dict, dict)
        self.assertEqual(as_dict["chunk_id"], "test_c001")
        self.assertEqual(as_dict["content"], "Nội dung kiểm thử")


if __name__ == "__main__":
    unittest.main()

