import unittest
from src.rag.chunker import (
    clean_markdown_syntax,
    VietnameseMarkdownCleaner,
    VietnameseHeadingPromoter,
    VietnameseSentenceSplitter,
    VietnameseHierarchicalChunker,
    ChunkerRegistry,
    ChunkerDispatcher,
    BaseChunker,
    Chunk,
)



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

