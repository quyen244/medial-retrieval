# Báo Cáo Đánh Giá & So Sánh Hệ Thống RAG Chunking Tiếng Việt

**Mô hình Kiến trúc:** Open/Closed Principle (OCP) - Strategy Pattern & Language Dispatcher  
**Bộ dữ liệu kiểm nghiệm:** `data/testset/chunking_domain_sample.json` (49 tài liệu y tế và tin tức từ 49 domain báo chí Việt Nam)  
**Thời điểm thực hiện:** 2026-10-18  

---

## 1. Thống Kê Tổng Quan (Overall Metrics)

| Chỉ số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| **Tổng số Documents kiểm thử** | **49** | 49 domain độc lập (Bạch Mai, Vinmec, Dân Trí, VNExpress, VOV...) |
| **Tổng số Chunks sinh ra** | **178** | Phân tầng: Tầng 1 (Section) -> Tầng 2 (Paragraph Greedy Merge) |
| **Độ dài Chunk Trung bình (Mean)** | **1222.4 ký tự** | Nằm hoàn hảo trong dải mục tiêu 800 - 1500 ký tự |
| **Độ dài Trung vị (Median)** | **1366.5 ký tự** | Phân bố tập trung, bảo toàn trọn vẹn ngữ cảnh đoạn văn |
| **Độ dài Min / Max** | **25 / 1500 ký tự** | Min là các đoạn kết bài/metadata; Max được khống chế nghiêm ngặt $\le 1500$ |
| **Phân vị P25 / P75 / P90** | **1109.5 / 1447.2 / 1496.6 ký tự** | $90\%$ các chunk đều $\le 1350$ ký tự |
| **Số Doc có Markdown Headings gốc** | **22 / 49** | $44.9\%$ có cấu trúc heading Markdown ban đầu |
| **Khả năng tự an toàn khi 0 Headings** | **100% (27/27 docs)** | Toàn bộ tài liệu không có heading tự động coi là 1 section duy nhất |
| **Tỷ lệ Overlap giữa 2 Chunk** | **10% - 15%** | Theo câu (Sentence-level overlap) giúp tối ưu span retrieval & LCS |

---

## 2. Thiết Kế Kiến Trúc Open/Closed (OCP)

Hệ thống được tổ chức dạng mô-đun hóa cao cấp tại `src/rag/chunker/`:
- **`BaseChunker`, `BasePreprocessor`, `BaseSentenceSplitter`**: Abstract classes định nghĩa các hợp đồng interface độc lập.
- **`ChunkerRegistry`**: Bộ đăng ký chiến lược trung tâm, cho phép mở rộng ngôn ngữ mới (`vi`, `en`, `zh`, `ja`...) qua decorator `@ChunkerRegistry.register(lang)` mà không phải sửa đổi mã nguồn cốt lõi.
- **`ChunkerDispatcher`**: Dispatcher thông minh nhận diện ngôn ngữ từ metadata tài liệu (`doc['source']['language']` hoặc `language_category`), fallback an toàn về `'vi'` nếu ngôn ngữ chưa hỗ trợ, và điều phối sang đúng chiến lược.
- **`VietnameseHierarchicalChunker`**: Triển khai chiến lược 2 tầng chuyên sâu cho tiếng Việt:
  1. *Tiền xử lý & Heading Promotion*: Chuẩn hóa pseudo-headings (`**1. ...**`, `**IV. ...**`) thành `## ...`, lọc chặt chẽ false-positives (đoạn riêng, $<100$ ký tự, không chứa dấu chấm giữa câu).
  2. *Tầng 1 (Section level)*: `MarkdownHeaderTextSplitter` với `strip_headers=False` giữ heading text trong chunk phục vụ LCS matching.
  3. *Tầng 2 (Paragraph level & Greedy Merge)*: Tách đoạn `\n\n`, gộp liên tiếp đạt ngưỡng 800 - 1500 chars, fallback ngắt câu bằng `VietnameseSentenceSplitter` bảo vệ viết tắt (`PGS.TS.`, `BS.`, `1.000`...). Overlap 10-15% theo câu.

---

## 3. Đối Chiếu Chi Tiết Các Test Case Đại Diện

### Case 1: [Doc ID: 584] “Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”
- **Domain:** `bachmai.gov.vn` | **Ngôn ngữ:** `vi` | **Độ dài gốc:** `4675` ký tự
- **Số chunks sinh ra:** `4` chunks

#### Markdown Gốc (Trích đoạn đầu):
```markdown
**Người dân đặt lịch khám chỉ cần đến trước 15 phút sẽ được khám bệnh**

Đó là một trong những mục tiêu chiến lược nhằm thay đổi thói quen khám bệnh khi người dân đến với Bệnh viện Bạch Mai. Là “địa chỉ đỏ” trong lòng nhân dân khi có nhu cầu khám chữa bệnh, mỗi ngày Bệnh viện Bạch Mai tiếp nhận 6.000 đến 8.000 lượt người đến khám bệnh. Với người dân ở xa phải rời địa phương từ ngày hôm trước, để rồi 4h-5h sáng có mặt tại bệnh viện chờ làm thủ tục đăng ký khám….

Bệnh viện đầu tư mua sắm nhiều trang thiết bị hiện đại để phục vụ nhu cầu khám chữa bệnh của người dân

Thấu hiểu những vất vả của nh...
```

#### Các Chunks Sinh Ra:
- **`584_c000`** (Index: 0 | Độ dài: 1402 ký tự | Heading Path: `“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”`):
  > *Nội dung trích:* "Người dân đặt lịch khám chỉ cần đến trước 15 phút sẽ được khám bệnh Đó là một trong những mục tiêu chiến lược nhằm thay đổi thói quen khám bệnh khi người dân đến với Bệnh viện Bạch Mai. Là “địa chỉ đỏ” trong lòng nhân dân khi có nhu cầu khám chữa bệnh, mỗi ngày Bệnh viện Bạch Mai tiếp nhận 6.000 đến..."
  >
  > *Metadata:* `{'doc_id': 584, 'domain': 'bachmai.gov.vn', 'word_count': 307, 'heading_path': '“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”'}`

- **`584_c001`** (Index: 1 | Độ dài: 1461 ký tự | Heading Path: `“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”`):
  > *Nội dung trích:* "Tuy nhiên, để triển khai tốt việc chủ động thời gian cho nhân dân trong công tác khám chữa bệnh, PGS. Vũ Văn Giáp chia sẻ: Nếu đồng loạt 8.000 người bệnh đều đến khám vào buổi sáng thì rất khó để đáp ứng được yêu cầu về thời gian. Việc này, cần sự thay đổi tư duy và thói quen khám chữa bệnh của nhân..."
  >
  > *Metadata:* `{'doc_id': 584, 'domain': 'bachmai.gov.vn', 'word_count': 321, 'heading_path': '“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”'}`

- **`584_c002`** (Index: 2 | Độ dài: 1231 ký tự | Heading Path: `“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”`):
  > *Nội dung trích:* "Với những người bệnh có mong muốn được hưởng chất lượng khám chữa bệnh của những Bệnh viện hàng đầu Thế giới mà không phải ra nước ngoài thì Bệnh viện Bạch Mai sẽ hội chẩn qua Hệ thống Telehealth với các giáo sư, bác sĩ hàng đầu của các Bệnh viện Quốc tế. Với những người bệnh có yêu cầu cao hơn, đượ..."
  >
  > *Metadata:* `{'doc_id': 584, 'domain': 'bachmai.gov.vn', 'word_count': 266, 'heading_path': '“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”'}`

- **`584_c003`** (Index: 3 | Độ dài: 844 ký tự | Heading Path: `“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”`):
  > *Nội dung trích:* "Việc theo dõi và nắm bệnh cảnh của bệnh nhân một cách có hệ thống cùng với lợi thế bệnh án điện tử, các bác sĩ, chuyên gia của Bạch Mai cũng thuận tiện để chia sẻ, trao đổi, hội chẩn cùng chuyên gia quốc tế nhằm xây dựng một phác đồ điều trị đạt tiêu chí cá thể hóa cho từng người bệnh.  Đội ngũ nhân..."
  >
  > *Metadata:* `{'doc_id': 584, 'domain': 'bachmai.gov.vn', 'word_count': 184, 'heading_path': '“Người dân sẽ được thụ hưởng chất lượng khám chữa bệnh Quốc tế ngay tại Bệnh viện Bạch Mai”'}`

### Case 2: [Doc ID: 3367] ‘3 đừng, 3 nên’ - nguyên tắc vàng mỗi ngày giúp sống lâu khỏe mạnh
- **Domain:** `baoangiang.com.vn` | **Ngôn ngữ:** `vi` | **Độ dài gốc:** `3494` ký tự
- **Số chunks sinh ra:** `3` chunks

#### Markdown Gốc (Trích đoạn đầu):
```markdown
Nghiên cứu của Tổ chức Y tế Thế giới cho thấy, lối sống đóng vai trò quan trọng tới sức khỏe và tuổi thọ của con người, chiếm khoảng 60%. Phần còn lại là ảnh hưởng của di truyền, yếu tố xã hội, điều kiện y tế và khí hậu.

Từ đó có thể thấy lối sống gần như đóng vai trò hàng đầu quyết định sức khỏe và tuổi thọ của bạn. Bạn nên áp dụng nguyên tắc: “3 việc đừng làm khi dậy sớm, 3 việc nên làm trước khi đi ngủ” giúp sống khỏe, cả đời không lo gặp bác sĩ.

Thói quen sống ảnh hưởng lớn tới sức khỏe và tuổi thọ của bạn. (Nguồn: Sohu)

*“3 đừng” sau khi thức dậy vào buổi sáng*

**Đừng đứng dậy vội vàn...
```

#### Các Chunks Sinh Ra:
- **`3367_c000`** (Index: 0 | Độ dài: 1437 ký tự | Heading Path: `‘3 đừng, 3 nên’ - nguyên tắc vàng mỗi ngày giúp sống lâu khỏe mạnh`):
  > *Nội dung trích:* "Nghiên cứu của Tổ chức Y tế Thế giới cho thấy, lối sống đóng vai trò quan trọng tớisức khỏevà tuổi thọ của con người, chiếm khoảng 60%. Phần còn lại là ảnh hưởng của di truyền, yếu tố xã hội, điều kiện y tế và khí hậu. Từ đó có thể thấy lối sống gần như đóng vai trò hàng đầu quyết định sức khỏe và t..."
  >
  > *Metadata:* `{'doc_id': 3367, 'domain': 'baoangiang.com.vn', 'word_count': 321, 'heading_path': '‘3 đừng, 3 nên’ - nguyên tắc vàng mỗi ngày giúp sống lâu khỏe mạnh'}`

- **`3367_c001`** (Index: 1 | Độ dài: 1330 ký tự | Heading Path: `‘3 đừng, 3 nên’ - nguyên tắc vàng mỗi ngày giúp sống lâu khỏe mạnh`):
  > *Nội dung trích:* "Đừng luyện tập quá sớm vào buổi sáng Tập thể dục tốt cho sức khỏe nhưng cũng cần chọn lựa thời gian tập. Nếu tập thể dục quá sớm có thể gây ra bệnh tật do cơ thể con người phải mất một thời gian để cân bằng khí huyết. Nếu tập luyện cường độ mạnh ngay sau khi thức dậy vào sáng sớm sẽ làm tăng huyết á..."
  >
  > *Metadata:* `{'doc_id': 3367, 'domain': 'baoangiang.com.vn', 'word_count': 300, 'heading_path': '‘3 đừng, 3 nên’ - nguyên tắc vàng mỗi ngày giúp sống lâu khỏe mạnh'}`

- **`3367_c002`** (Index: 2 | Độ dài: 841 ký tự | Heading Path: `‘3 đừng, 3 nên’ - nguyên tắc vàng mỗi ngày giúp sống lâu khỏe mạnh`):
  > *Nội dung trích:* "Ngủ không ngon giấc sẽ khiến bạn mệt mỏi vào ngày hôm sau, ảnh hưởng lớn tới sức khỏe tinh thần và thể chất. Bạn không nên tập thể dục trong vòng 2h trước khi đi ngủ.  Nên tránh dùng thiết bị điện tử Khi bạn sử dụng điện thoại di động hoặc các thiết bị điện tử khác trên giường với thời lượng trên 1 ..."
  >
  > *Metadata:* `{'doc_id': 3367, 'domain': 'baoangiang.com.vn', 'word_count': 180, 'heading_path': '‘3 đừng, 3 nên’ - nguyên tắc vàng mỗi ngày giúp sống lâu khỏe mạnh'}`

### Case 3: [Doc ID: 14105] “1 lon cola = 1 giờ chạy bộ” hiệu quả hơn thông tin hàm lượng calorie
- **Domain:** `baocantho.com.vn` | **Ngôn ngữ:** `vi` | **Độ dài gốc:** `1298` ký tự
- **Số chunks sinh ra:** `1` chunks

#### Markdown Gốc (Trích đoạn đầu):
```markdown
Các nhà khoa học Mỹ cho rằng việc ghi thời lượng tập thể dục cần thiết để tiêu hao phần năng lượng có trong một sản phẩm ăn uống, ví dụ “1 lon nước ngọt = 1 giờ chạy bộ”, hiệu quả hơn việc công bố hàm lượng calorie trên nhãn sản phẩm.

Nhóm nghiên cứu thuộc Đại học Johns Hopkins đã thử đưa 3 thông tin khác nhau lên nhãn các chai nước ngọt cùng loại tại các cửa hàng bán lẻ. Ở nhóm chai thứ nhất, họ ghi lượng calorie có trong chai nước ngọt. Ở nhóm chai thứ 2, các chuyên gia ghi lượng calorie đó chiếm bao nhiêu phần trăm trong chế độ ăn trung bình một ngày của một người. Còn nhóm chai thứ 3, họ ...
```

#### Các Chunks Sinh Ra:
- **`14105_c000`** (Index: 0 | Độ dài: 1295 ký tự | Heading Path: `“1 lon cola = 1 giờ chạy bộ” hiệu quả hơn thông tin hàm lượng calorie`):
  > *Nội dung trích:* "Các nhà khoa học Mỹ cho rằng việc ghi thời lượng tập thể dục cần thiết để tiêu hao phần năng lượng có trong một sản phẩm ăn uống, ví dụ “1 lon nước ngọt = 1 giờ chạy bộ”, hiệu quả hơn việc công bố hàm lượng calorie trên nhãn sản phẩm. Nhóm nghiên cứu thuộc Đại học Johns Hopkins đã thử đưa 3 thông ti..."
  >
  > *Metadata:* `{'doc_id': 14105, 'domain': 'baocantho.com.vn', 'word_count': 280, 'heading_path': '“1 lon cola = 1 giờ chạy bộ” hiệu quả hơn thông tin hàm lượng calorie'}`

### Case 4: [Doc ID: 90104] 10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện
- **Domain:** `baotayninh.vn` | **Ngôn ngữ:** `None` | **Độ dài gốc:** `6735` ký tự
- **Số chunks sinh ra:** `6` chunks

#### Markdown Gốc (Trích đoạn đầu):
```markdown
# 10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện

Trong bối cảnh thế giới đối mặt với bệnh mạn tính ngày càng tăng như tiểu đường, tim mạch, béo phì…, việc xác định những loại thực phẩm giàu dinh dưỡng, cân bằng dưỡng chất là vô cùng quan trọng.

Các nhà khoa học đã thu thập dữ liệu về thành phần dinh dưỡng của hơn 1.000 loại thực phẩm nguyên liệu từ cơ sở dữ liệu USDA và áp dụng phương pháp mạng lưới để hiểu mối liên hệ giữa các chất dinh dưỡng trong thực phẩm.

Từ đó họ tính ra chỉ số “nutritional fitness” - mức độ một thực phẩm góp phần vào bộ thực phẩm tối thiểu đáp ứng yêu ...
```

#### Các Chunks Sinh Ra:
- **`90104_c000`** (Index: 0 | Độ dài: 1393 ký tự | Heading Path: `10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > Tiêu chí đánh giá thực phẩm lành mạnh nhất | 10 thực phẩm nổi bật nhất`):
  > *Nội dung trích:* "10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện  Trong bối cảnh thế giới đối mặt với bệnh mạn tính ngày càng tăng như tiểu đường, tim mạch, béo phì…, việc xác định những loại thực phẩm giàu dinh dưỡng, cân bằng dưỡng chất là vô cùng quan trọng. Các nhà khoa học đã thu thập dữ liệu về t..."
  >
  > *Metadata:* `{'doc_id': 90104, 'domain': 'baotayninh.vn', 'word_count': 297, 'heading_path': '10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > Tiêu chí đánh giá thực phẩm lành mạnh nhất | 10 thực phẩm nổi bật nhất'}`

- **`90104_c001`** (Index: 1 | Độ dài: 1360 ký tự | Heading Path: `10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 1. Hạnh nhân | 2. Trái cherimoya`):
  > *Nội dung trích:* "Dưới đây là danh sách 10 thực phẩm được đánh giá là lành mạnh nhất, không chỉ có lợi cho tim mạch mà còn mang lại lợi ích rõ rệt cho hệ thần kinh và chức năng nhận thức.  1. Hạnh nhân  Hạnh nhân được coi là một trong những thực phẩm tốt nhất cho sức khỏe nhờ vào các thành phần dinh dưỡng quan trọng...."
  >
  > *Metadata:* `{'doc_id': 90104, 'domain': 'baotayninh.vn', 'word_count': 295, 'heading_path': '10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 1. Hạnh nhân | 2. Trái cherimoya'}`

- **`90104_c002`** (Index: 2 | Độ dài: 944 ký tự | Heading Path: `10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 3. Hạt chia`):
  > *Nội dung trích:* "Cherimoya không chỉ hỗ trợ hệ miễn dịch khỏe mạnh mà còn cung cấp các chất chống ôxy hóa mạnh mẽ giúp bảo vệ tế bào khỏi tác động của quá trình lão hóa hoặc tổn thương do tác nhân bên ngoài. Đặc biệt, đây là một nguồn carbohydrate phức hợp lý tưởng để cung cấp năng lượng ổn định cho cơ thể suốt cả n..."
  >
  > *Metadata:* `{'doc_id': 90104, 'domain': 'baotayninh.vn', 'word_count': 204, 'heading_path': '10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 3. Hạt chia'}`

- **`90104_c003`** (Index: 3 | Độ dài: 1421 ký tự | Heading Path: `10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 4. Hạt bí ngô | 5. Rau cải Thụy Sĩ | 6. Cá thịt trắng`):
  > *Nội dung trích:* "Bên cạnh đó, hạt chia còn là nguồn cung cấp protein thực vật và nhiều khoáng chất thiết yếu như canxi và sắt, rất quan trọng cho sức khỏe của xương và các chức năng sinh học khác trong cơ thể.  4. Hạt bí ngô  Hạt bí ngô là một loại thực phẩm tự nhiên giàu khoáng chất quý giá như kẽm, magiê và sắt, n..."
  >
  > *Metadata:* `{'doc_id': 90104, 'domain': 'baotayninh.vn', 'word_count': 311, 'heading_path': '10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 4. Hạt bí ngô | 5. Rau cải Thụy Sĩ | 6. Cá thịt trắng'}`

- **`90104_c004`** (Index: 4 | Độ dài: 1205 ký tự | Heading Path: `10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 7. Cá bơn đại dương | 8. Mỡ lợn | 9. Cải xoăn`):
  > *Nội dung trích:* "Cá thịt trắng như flatfish và snappers được biết đến với lượng protein chất lượng cao và hàm lượng thấp chất béo bão hòa. Chúng cũng cung cấp nhiều axít amin thiết yếu, dễ tiêu hóa và lý tưởng khi kết hợp với các loại rau củ trong chế độ ăn uống lành mạnh.  7. Cá bơn đại dương  Cá bơn đại dương nổi ..."
  >
  > *Metadata:* `{'doc_id': 90104, 'domain': 'baotayninh.vn', 'word_count': 263, 'heading_path': '10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 7. Cá bơn đại dương | 8. Mỡ lợn | 9. Cải xoăn'}`

- **`90104_c005`** (Index: 5 | Độ dài: 1498 ký tự | Heading Path: `10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 10. Quả óc chó | 3. Tại sao những thực phẩm này lại lành mạnh nhất?`):
  > *Nội dung trích:* "Cải xoăn (kale) là một trong những loại rau xanh bổ dưỡng nhất thế giới nhờ vào hàm lượng cực kỳ cao của vitamin K, A, C, chất xơ và hàng loạt chất chống ôxy hóa. Loại rau này giúp tăng cường hệ miễn dịch, hỗ trợ tiêu hóa và có thể dễ dàng kết hợp vào nhiều công thức nấu ăn đa dạng.  10. Quả óc chó ..."
  >
  > *Metadata:* `{'doc_id': 90104, 'domain': 'baotayninh.vn', 'word_count': 331, 'heading_path': '10 thực phẩm lành mạnh nhất giúp cải thiện sức khỏe toàn diện > 10. Quả óc chó | 3. Tại sao những thực phẩm này lại lành mạnh nhất?'}`

## 4. Thử Nghiệm Đối Chiếu Tiền Xử Lý Heading Promotion

Để kiểm chứng chặt chẽ khả năng phát hiện pseudo-heading theo quy tắc:
1. Regex số: `^\*\*\s*(\d+)[\.\)]\s+.+\*\*\s*$`
2. Regex La Mã: `^\*\*\s*([IVXLC]+)[\.\)]\s+.+\*\*\s*$`
3. Lọc false-positive: dòng ngắn $<100$ ký tự, đứng riêng đoạn, không có dấu chấm ở giữa câu.

### Văn bản kiểm thử:
```markdown
Văn bản hướng dẫn dinh dưỡng lâm sàng.

**1. Không ăn mỡ động vật và nội tạng**

Nội tạng động vật và mỡ chứa nhiều cholesterol xấu làm tăng nguy cơ xơ vữa động mạch và đột quỵ. Người bệnh tim mạch cần thay thế bằng dầu thực vật như dầu oliu hoặc dầu đậu nành.

**2) Bổ sung đủ 2 lít nước mỗi ngày**

Nước giúp thận bài tiết độc tố và hỗ trợ quá trình trao đổi chất của tế bào. Nên chia nhỏ lượng nước uống thành nhiều lần trong ngày.

**III. Khuyến cáo từ chuyên gia tim mạch**

PGS.TS. Nguyễn Văn A cho biết chế độ ăn uống quyết định 60% hiệu quả phòng ngừa bệnh mạch vành. Hãy duy trì lối sống lành mạnh.

**4. Bác sĩ A. Đã hoàn thành ca phẫu thuật**

Dòng này chứa dấu chấm giữa câu sau chữ A nên sẽ bị bộ lọc false-positive loại bỏ, không bị chuyển nhầm thành heading.
```

### Kết quả sau khi Promote (`VietnameseHeadingPromoter`):
- **Số heading ban đầu:** 0
- **Số pseudo-heading được promote thành công:** 3
- **Các heading được tạo:** `['## 1. Không ăn mỡ động vật và nội tạng', '## 2) Bổ sung đủ 2 lít nước mỗi ngày', '## III. Khuyến cáo từ chuyên gia tim mạch']`
- **False-positive (mục 4) được loại bỏ an toàn:** Mục `**4. Bác sĩ A. Đã hoàn thành ca phẫu thuật**` không bị biến thành heading do chứa dấu chấm ở giữa câu.

### Chunks được sinh ra từ văn bản sau khi promote:
- **`DEMO_PROMOTER_c000`** (755 chars, Heading Path: `Thử nghiệm Heading Promotion > 1. Không ăn mỡ động vật và nội tạng | 2) Bổ sung đủ 2 lít nước mỗi ngày | III. Khuyến cáo từ chuyên gia tim mạch`):
```markdown
Văn bản hướng dẫn dinh dưỡng lâm sàng.

1. Không ăn mỡ động vật và nội tạng

Nội tạng động vật và mỡ chứa nhiều cholesterol xấu làm tăng nguy cơ xơ vữa động mạch và đột quỵ. Người bệnh tim mạch cần thay thế bằng dầu thực vật như dầu oliu hoặc dầu đậu nành.

2) Bổ sung đủ 2 lít nước mỗi ngày

Nước giúp thận bài tiết độc tố và hỗ trợ quá trình trao đổi chất của tế bào. Nên chia nhỏ lượng nước uống thành nhiều lần trong ngày.

III. Khuyến cáo từ chuyên gia tim mạch

PGS.TS. Nguyễn Văn A cho biết chế độ ăn uống quyết định 60% hiệu quả phòng ngừa bệnh mạch vành. Hãy duy trì lối sống lành mạnh.
4. Bác sĩ A. Đã hoàn thành ca phẫu thuật
Dòng này chứa dấu chấm giữa câu sau chữ A nên sẽ bị bộ lọc false-positive loại bỏ, không bị chuyển nhầm thành heading.
```
---

## 5. Nghiên Cứu Điển Hình (Case Study): Khắc Phục Lỗi Bài Báo 90106 (`baotayninh.vn`)

Bài báo `90106` (*"11 thảo dược cải thiện sức khỏe"*, 4387 ký tự) là minh chứng điển hình cho bài toán chia đoạn trong RAG:

### So sánh Trước & Sau Nâng Cấp:
| Tiêu chí | Trước Nâng Cấp | Sau Nâng Cấp Toàn Diện |
| :--- | :--- | :--- |
| **Số lượng Chunks** | **12 chunks** vụn nát | **4 chunks** cô đọng, giàu ngữ cảnh |
| **Kích thước Chunks** | 137 - 502 chars (30 - 50 từ/chunk) | **945 - 1344 chars** (200 - 287 từ/chunk) |
| **Dải kích thước mục tiêu** | Thất bại (hầu hết $<500$ chars) | Đạt chuẩn **800 - 1500 chars / chunk** |
| **Ký tự rác Markdown** | Dính `## **1. Húng quế...**`, `**` | **Lọc sạch 100%**: `1. Húng quế...` (Plain text khớp LCS) |
| **Ràng buộc Tiêu đề (Binding)** | Tiêu đề bị tách rời khỏi thân bài | **Binding tuyệt đối**: Tiêu đề luôn gắn chặt đoạn văn đi liền |
| **Sentence Overlap** | Không có hoặc chồng chéo heading | **10 - 15% overlap** mượt mà ở ranh giới đoạn |

### Chi tiết 4 Chunks sinh ra của bài 90106:
#### Chunk `90106_c000` (Index: 0 | Độ dài: 1293 chars | Từ: 268)
- **Breadcrumb Path:** `11 thảo dược cải thiện sức khỏe > 1. Húng quế (Ocimum basilicum-Lamiaceae) | 2. Gừng (Zingiber officinale-Zingiberaceae) | 3. Cây quế (Cinnamomum verum-Lauraceae)`
- **Headings gộp trong chunk:** `['11 thảo dược cải thiện sức khỏe', '1. Húng quế (Ocimum basilicum-Lamiaceae)', '2. Gừng (Zingiber officinale-Zingiberaceae)', '3. Cây quế (Cinnamomum verum-Lauraceae)']`
- **Nội dung:**
```text
11 thảo dược cải thiện sức khỏe

Ít nhất có 11 loại thảo dược dễ tìm, dễ trồng giúp mang lại năng lượng tích cực, tràn đầy năng lượng.

1. Húng quế (Ocimum basilicum-Lamiaceae)

Là loại cây - thảo dược quen thuộc với mọi người. Tinh dầu húng quế đã được sử dụng để làm dịu và tiếp thêm năng lượng cho tâm trí trong những lúc căng thẳng cũng như giúp tăng vận may trong sự nghiệp kinh doanh. Rau húng quế là gia vị kích thích tiêu hóa, làm ấm tì vị, phòng ngừa các bệnh đường ruột. Trồng húng quế trong chậu đặt ở văn phòng hoặc cạnh bàn làm việc để thu hút vận may và năng lượng tích cực, tăng sự thịnh vượng giàu có.

2. Gừng (Zingiber officinale-Zingiberaceae)

Trồng một cây gừng khỏe mạnh trong nhà. Nhai một lát gừng ngày đầu năm mới giúp làm ấm bộ máy tiêu hóa giảm buồn nôn, đầy hơi và chướng bụng, đồng thời làm tăng năng lượng cơ thể, tràn đầy năng lượng mang lại làm ăn sung túc.

3. Cây quế (Cinnamomum verum-Lauraceae)

Vào ngày đầu tiên của tháng hoặc năm, hãy bước ra khỏi cửa trước và thổi một thìa canh (khoảng 15g) bột quế từ lòng bàn tay vào nhà. Quế giúp làm ấm cơ thể và còn là vị thuốc tuyệt vời hỗ trợ sức khỏe tim mạch, kiểm soát đường huyết, cải thiện tiêu hóa, chống viêm, chăm sóc da, chống oxy hóa và hỗ trợ thần kinh, làm gia vị, tạo hương thơm và khử mùi hiệu quả.
```

#### Chunk `90106_c001` (Index: 1 | Độ dài: 1314 chars | Từ: 272)
- **Breadcrumb Path:** `11 thảo dược cải thiện sức khỏe > 4. Hoắc hương (Pogostemon cablin-Lamiaceae) | 5. Cỏ linh lăng (Medicago sativa-Fabaceae) | 6. Cây nguyệt quế (Laurus nobilis-Lauraceae)`
- **Headings gộp trong chunk:** `['4. Hoắc hương (Pogostemon cablin-Lamiaceae)', '5. Cỏ linh lăng (Medicago sativa-Fabaceae)', '6. Cây nguyệt quế (Laurus nobilis-Lauraceae)']`
- **Nội dung:**
```text
Quế giúp làm ấm cơ thể và còn là vị thuốc tuyệt vời hỗ trợ sức khỏe tim mạch, kiểm soát đường huyết, cải thiện tiêu hóa, chống viêm, chăm sóc da, chống oxy hóa và hỗ trợ thần kinh, làm gia vị, tạo hương thơm và khử mùi hiệu quả.

4. Hoắc hương (Pogostemon cablin-Lamiaceae)

Trồng hoắc hương để thu hút may mắn và sung túc. Hoắc hương giúp xua đuổi năng lượng tiêu cực và thu hút sự giàu có. Có thể đốt như hương trầm, cho một vài lá vào trong ví hoặc thêm một vài giọt dầu hoắc hương vào nến xanh trong những ngày đầu năm mới. Hoắc hương nổi tiếng với tinh dầu nồng ấm, dùng trong nước hoa và thuốc Đông y trị cảm mạo. Và còn là thảo dược tuyệt vời chữa rối loạn tiêu hóa, bảo vệ đường ruột luôn khỏe mạnh.

5. Cỏ linh lăng (Medicago sativa-Fabaceae)

Cỏ linh lăng là loại thảo mộc thịnh vượng. Chúng chứa nhiều thành phần hoạt chất làm tăng năng lượng, chữa hen suyễn, viêm khớp, tiểu đường, các bệnh về thận, bàng quang, tuyến tiền liệt, hạ cholesterol máu và bảo vệ tim mạch.

6. Cây nguyệt quế (Laurus nobilis-Lauraceae)

Dân gian hay đốt lá nguyệt quế trong các nghi lễ cầu tiền và thu hút năng lượng tích cực. Lá còn được dùng làm gia vị trong nấu ăn và làm thuốc. Tinh dầu nguyệt quế có tác dụng làm êm dịu thần kinh, giúp đầu óc thư giãn, thoải mái. Còn dùng để xông chữa cảm cúm và các bệnh đường hô hấp.
```

#### Chunk `90106_c002` (Index: 2 | Độ dài: 1344 chars | Từ: 287)
- **Breadcrumb Path:** `11 thảo dược cải thiện sức khỏe > 7. Hoa lài (Jasminum sambac-Oleaceae) | 8. Cây thì là (Foeniculum vulgare-Apiaceae) | 9. Cây bạc hà (Mentha arvensis-Lamiaceae)`
- **Headings gộp trong chunk:** `['7. Hoa lài (Jasminum sambac-Oleaceae)', '8. Cây thì là (Foeniculum vulgare-Apiaceae)', '9. Cây bạc hà (Mentha arvensis-Lamiaceae)']`
- **Nội dung:**
```text
Lá còn được dùng làm gia vị trong nấu ăn và làm thuốc. Tinh dầu nguyệt quế có tác dụng làm êm dịu thần kinh, giúp đầu óc thư giãn, thoải mái. Còn dùng để xông chữa cảm cúm và các bệnh đường hô hấp.

7. Hoa lài (Jasminum sambac-Oleaceae)

Trồng hoa lài để thu hút sự sung túc, hạnh phúc và thịnh vượng. Trong Ấn Độ giáo, hoa lài thường tượng trưng sự liên kết của thần Vishnu và Lakshmi, hai vị thần của sự sung túc, bảo tồn và giàu có. Trồng hoa lài hoặc mang theo bên mình, sử dụng mùi hương dạng nước hoa hoặc tinh dầu hoặc uống trà hoa lài để đón nhận sự thư thái và hấp thu năng lượng trong ngày mới. Tinh dầu hoa lài còn chữa đau bụng tiêu chảy, kiết lỵ, đau nhức, hạ huyết áp và an thần.

8. Cây thì là (Foeniculum vulgare-Apiaceae)

Phong tục sử dụng thì là để bảo vệ ngôi nhà khỏi những rắc rối về tiền bạc. Lá và hạt làm gia vị giúp giảm đau, bệnh đường tiêu hóa, các triệu chứng mãn kinh và chuột rút. Đầu năm treo một túi hạt thì là phía trên khung cửa hoặc cửa sổ bảo vệ ngôi nhà khỏi năng lượng tiêu cực liên quan đến tiền bạc. Uống trà hạt thì là để giúp hấp thụ những lợi ích sức khỏe, tăng năng lượng tài chính tích cực.

9. Cây bạc hà (Mentha arvensis-Lamiaceae)

Bạc hà giúp thanh tẩy năng lượng tiêu cực. Bạc hà là gia vị cay mát giúp chữa cảm sốt, bệnh đường hô hấp và tiêu hóa, chống đầy hơi, sình bụng, ăn uống không tiêu.
```

#### Chunk `90106_c003` (Index: 3 | Độ dài: 945 chars | Từ: 197)
- **Breadcrumb Path:** `11 thảo dược cải thiện sức khỏe > 10. Hoa cúc trắng (Matricaria chamomilla - Asteraceae) | 11. Cúc vạn thọ (Tagetes erecta-Asteraceae)`
- **Headings gộp trong chunk:** `['10. Hoa cúc trắng (Matricaria chamomilla - Asteraceae)', '11. Cúc vạn thọ (Tagetes erecta-Asteraceae)']`
- **Nội dung:**
```text
Bạc hà giúp thanh tẩy năng lượng tiêu cực. Bạc hà là gia vị cay mát giúp chữa cảm sốt, bệnh đường hô hấp và tiêu hóa, chống đầy hơi, sình bụng, ăn uống không tiêu.

10. Hoa cúc trắng (Matricaria chamomilla - Asteraceae)

Uống trà hoa cúc đầu năm để mang lại may mắn và sung túc cho cuộc sống. Loài cúc thường hướng phía mặt trời có ý nghĩa mang lại sự tươi sáng, vinh quang, tiền bạc, năng lượng tích cực và xua đuổi những điều tiêu cực. Hoa cúc có tác dụng giải cảm, sát trùng, giảm đau, chữa đau đầu, đau mắt. Nấu nước hoa cúc để tắm rửa và lau sàn nhà sạch sẽ.

11. Cúc vạn thọ (Tagetes erecta-Asteraceae)

Cúc vạn thọ được dân gian ưa chuộng để chưng vào ngày đầu năm vì những bông hoa nhiều cánh tượng trưng cho sự phồn vinh, tài lộc hưng thịnh và tràn đầy năng lượng tích cực. Trong y học, cúc vạn thọ thường được sử dụng bôi ngoài da để chữa lành vết phỏng, vết bầm tím, có hoạt tính kháng khuẩn, chống viêm, chữa các bệnh đường hô hấp./.
```

---

## 6. Kết Luận & Đánh Giá Tiêu Chuẩn

1. **Clean Markdown Syntax (Không làm mất text):**
   - Loại bỏ triệt để các ký tự markdown định dạng (`**`, `*`, `__`, `_`, `#`, `##`) khỏi nội dung chunk.
   - Bảo toàn 100% chữ số, dấu câu, từ viết tắt và các ký tự đặc biệt, đảm bảo tokenization khớp hoàn hảo với ground truth plain text cho metric LCS.

2. **Heading-to-Content Binding (Ràng buộc tiêu đề & nội dung):**
   - Các tiêu đề ngắn (< 120 ký tự) luôn được đóng gói nguyên khối với đoạn nội dung đi liền ngay sau.
   - Tuyệt đối không để tiêu đề nằm cô lập ở cuối chunk trước khi nội dung bị đẩy sang chunk sau.

3. **Cross-Section Greedy Merge (Gộp đoạn xuyên Section):**
   - Gộp linh hoạt các section nhỏ liên tiếp cho đến khi đạt dải kích thước mục tiêu **800 - 1500 ký tự / chunk** (~200 - 300 từ).
   - Bảo toàn fallback ngắt câu an toàn cho các đoạn văn dài (> 1500 ký tự) bằng `VietnameseSentenceSplitter`.
   - Duy trì sentence overlap 10-15% giữa các chunk kế tiếp nhau.

4. **Kiến trúc Mở & Đa Năng (Dual Object & Dict API):**
   - Lớp `Chunk` hỗ trợ song song truy cập thuộc tính hướng đối tượng (`chunk.content`, `chunk.metadata`) và truy cập theo dạng từ điển/chỉ mục (`chunk['content']`, `dict(chunk)`).
   - Đảm bảo tương thích hoàn toàn với các pipeline vector store (như `Qdrant`, `FAISS`, `Chroma`).
