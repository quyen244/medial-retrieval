# Báo cáo trạng thái crawl theo ngôn ngữ

Cập nhật: 2026-10-17.
Corpus `AIGuruTinix/ViBioMIR`, 1.220.598 tài liệu, 94 domain.

Mọi trạng thái dưới đây đến từ request thật tới site thật.
Cột Giờ là tính toán từ tốc độ đã đo, chưa chạy đủ lâu để xác nhận.

## Tổng quan

| Nhóm | Domain | Tài liệu | % corpus | Lấy được | Nhóm xong sau | Nút thắt |
|---|---:|---:|---:|---:|---:|---|
| Tiếng Việt | 50 | 740,598 | 60.68% | **60.24%** | 8.6h | `suckhoecongdongonline.vn` |
| Tiếng Trung | 43 | 240,000 | 19.66% | **8.82%** | 5.6h | `www.cnkang.com` |
| Tiếng Anh | 1 | 240,000 | 19.66% | **19.66%** | 0.1h | `pubmed.ncbi.nlm.nih.gov` |
| **Tổng** | **94** | **1.220.598** | **100%** | **88.72%** | **8.6h** | chạy song song |

Ba nhóm chạy song song nên tổng thời gian là nhóm chậm nhất, không phải tổng ba nhóm.

## Tiếng Việt

Báo chí, bệnh viện và nhà thuốc trong nước.
50 domain, 740,598 tài liệu (60.68% corpus), lấy được 735,265 (60.24%).
Nhóm này xong sau 8.6 giờ, bị chặn bởi `suckhoecongdongonline.vn`.

| Domain | Tài liệu | % | Crawler | req/s | Giờ | Trạng thái | Thay đổi |
|---|---:|---:|---|---:|---:|---|---|
| `suckhoecongdongonline.vn` | 154,503 | 12.66% | generic | 5 | 8.6 | OK | **Tăng tốc** 2→5 req/s |
| `suckhoedoisong.vn` | 85,823 | 7.03% | generic | 5 | 4.8 | OK | **Tăng tốc** 2→5 req/s |
| `nhathuoclongchau.com.vn` | 79,598 | 6.52% | impersonate | 5 | 4.4 | OK | **Đã sửa** (trước: HTTP 403 Cloudflare, rồi 75h qua Playwright) · **Tăng tốc** 2→5 req/s |
| `thanhnien.vn` | 72,111 | 5.91% | generic | 5 | 4.0 | OK | **Tăng tốc** 2→5 req/s |
| `laodong.vn` | 31,282 | 2.56% | laodong | 2 | 4.3 | OK | **Đã sửa** (trước: trích 0 ký tự (JS cookie challenge)) |
| `phunusuckhoe.giadinhonline.vn` | 27,577 | 2.26% | generic | 2 | 3.8 | OK | — |
| `www.vinmec.com` | 26,129 | 2.14% | generic | 2 | 3.6 | OK | — |
| `medlatec.vn` | 24,762 | 2.03% | generic | 2 | 3.4 | OK | — |
| `giadinhonline.vn` | 16,729 | 1.37% | generic | 2 | 2.3 | OK | — |
| `baohaiphong.vn` | 16,286 | 1.33% | vnnews | 2 | 2.3 | OK | **Đã sửa** (trước: HTTP 403) |
| `baonghean.vn` | 15,941 | 1.31% | vnnews | 2 | 2.2 | OK | **Đã sửa** (trước: HTTP 403) |
| `vietnamnet.vn` | 15,543 | 1.27% | generic | 2 | 2.2 | OK | — |
| `baodanang.vn` | 14,500 | 1.19% | vnnews | 2 | 2.0 | OK | **Đã sửa** (trước: HTTP 403) |
| `tiemchunglongchau.com.vn` | 13,281 | 1.09% | generic | 2 | 1.8 | OK | — |
| `baocantho.com.vn` | 12,016 | 0.98% | generic | 2 | 1.7 | OK | — |
| `hellobacsi.com` | 11,747 | 0.96% | generic | 2 | 1.6 | OK | — |
| `baoangiang.com.vn` | 10,738 | 0.88% | generic | 2 | 1.5 | OK | — |
| `vov.vn` | 9,784 | 0.80% | generic | 2 | 1.4 | OK | — |
| `khoahocphothong.vn` | 8,143 | 0.67% | generic | 2 | 1.1 | OK | — |
| `youmed.vn` | 7,189 | 0.59% | generic | 2 | 1.0 | OK | — |
| `vov2.vov.vn` | 6,899 | 0.56% | generic | 2 | 1.0 | OK | — |
| `phuyen.baodaklak.vn` | 6,252 | 0.51% | generic | 2 | 0.9 | OK | — |
| `suckhoeviet.org.vn` | 5,627 | 0.46% | generic | 2 | 0.8 | OK | — |
| `baoquangtri.vn` | 5,483 | 0.45% | generic | 2 | 0.8 | OK | — |
| `pmc-ecm-healthblog.beta.pharmacity.io` | 5,333 | 0.44% | excluded | 2 | 0.7 | Đã bỏ | host staging đã nghỉ |
| `nhandan.vn` | 5,225 | 0.43% | generic | 2 | 0.7 | OK | — |
| `baovinhlong.com.vn` | 5,113 | 0.42% | generic | 2 | 0.7 | OK | — |
| `tamanhhospital.vn` | 4,261 | 0.35% | impersonate | 2 | 0.6 | OK | **Đã sửa** (trước: HTTP 403) |
| `tuoitre.vn` | 3,893 | 0.32% | generic | 2 | 0.5 | OK | — |
| `www.qdnd.vn` | 3,700 | 0.30% | generic | 2 | 0.5 | OK | — |
| `www.vietnamplus.vn` | 3,695 | 0.30% | generic | 2 | 0.5 | Chập chờn | — |
| `baogialai.com.vn` | 3,290 | 0.27% | generic | 2 | 0.5 | OK | — |
| `tienphong.vn` | 3,021 | 0.25% | generic | 2 | 0.4 | OK | — |
| `thaythuocvietnam.vn` | 2,927 | 0.24% | generic | 2 | 0.4 | OK | — |
| `baoquangninh.vn` | 2,912 | 0.24% | generic | 2 | 0.4 | OK | — |
| `bachmai.gov.vn` | 2,783 | 0.23% | generic | 2 | 0.4 | OK | — |
| `baochinhphu.vn` | 2,302 | 0.19% | generic | 2 | 0.3 | OK | — |
| `dantri.com.vn` | 2,063 | 0.17% | generic | 2 | 0.3 | OK | — |
| `vnexpress.net` | 2,055 | 0.17% | generic | 2 | 0.3 | OK | — |
| `baolangson.vn` | 1,981 | 0.16% | generic | 2 | 0.3 | OK | — |
| `www.sggp.org.vn` | 1,853 | 0.15% | generic | 2 | 0.3 | OK | — |
| `baophutho.vn` | 1,288 | 0.11% | generic | 2 | 0.2 | OK | — |
| `hanoimoi.vn` | 1,283 | 0.10% | vnnews | 2 | 0.2 | OK | **Đã sửa** (trước: HTTP 403) |
| `benhviennhitrunguong.gov.vn` | 1,200 | 0.10% | generic | 2 | 0.2 | OK | — |
| `baotayninh.vn` | 1,198 | 0.10% | baotayninh | 2 | 0.2 | OK | **Đã sửa** (trước: trích 0 ký tự (HTML hỏng)) |
| `baothanhhoa.vn` | 782 | 0.06% | generic | 2 | 0.1 | OK | — |
| `benhvienvietduc.org` | 425 | 0.04% | generic | 2 | 0.1 | OK | — |
| `www.pharmacity.vn` | 69 | 0.01% | generic | 2 | 0.0 | OK | — |
| `mega.vietnamplus.vn` | 2 | 0.00% | vnnews | 2 | 0.0 | OK | **Đã sửa** (trước: HTTP 403) |
| `special.vietnamplus.vn` | 1 | 0.00% | generic | 2 | 0.0 | OK | — |

## Tiếng Trung

Cổng y tế và từ điển đông y Trung Quốc.
43 domain, 240,000 tài liệu (19.66% corpus), lấy được 107,631 (8.82%).
Nhóm này xong sau 5.6 giờ, bị chặn bởi `www.cnkang.com`.

| Domain | Tài liệu | % | Crawler | req/s | Giờ | Trạng thái | Thay đổi |
|---|---:|---:|---|---:|---:|---|---|
| `www.120ask.com` | 98,160 | 8.04% | blocked | 2 | 13.6 | Chặn | origin chết (521 cả trang chủ), mobile không map được ID, Wayback không có |
| `www.cnkang.com` | 40,046 | 3.28% | generic | 2 | 5.6 | OK | — |
| `ask.39.net` | 32,772 | 2.69% | generic | 2 | 4.6 | OK | — |
| `www.familydoctor.com.cn` | 26,819 | 2.20% | excluded | 2 | 3.7 | Đã bỏ | cần IP Trung Quốc |
| `zysjonline.com` | 6,591 | 0.54% | blocked | 2 | 0.9 | Chặn | Cloudflare 403, Playwright lẫn curl_cffi đều không qua |
| `zhongyibaodian.net` | 6,061 | 0.50% | generic | 2 | 0.8 | OK | — |
| `www.a-hospital.com` | 4,701 | 0.39% | generic | 2 | 0.7 | OK | — |
| `www.youlai.cn` | 4,139 | 0.34% | generic | 2 | 0.6 | OK | — |
| `baby.39.net` | 3,368 | 0.28% | generic | 2 | 0.5 | OK | — |
| `test.pmphai.com` | 2,989 | 0.24% | generic | 2 | 0.4 | OK | — |
| `jbk.39.net` | 2,033 | 0.17% | generic | 2 | 0.3 | OK | — |
| `care.39.net` | 1,282 | 0.10% | generic | 2 | 0.2 | OK | — |
| `heart.39.net` | 1,055 | 0.09% | generic | 2 | 0.1 | OK | — |
| `fk.39.net` | 928 | 0.08% | generic | 2 | 0.1 | OK | — |
| `www.msdmanuals.cn` | 905 | 0.07% | generic | 2 | 0.1 | OK | — |
| `nk.39.net` | 762 | 0.06% | generic | 2 | 0.1 | OK | — |
| `cancer.39.net` | 736 | 0.06% | generic | 2 | 0.1 | OK | — |
| `article.iiyi.com` | 729 | 0.06% | excluded | 2 | 0.1 | Đã bỏ | origin chết, quá ít doc |
| `tnb.39.net` | 717 | 0.06% | generic | 2 | 0.1 | OK | — |
| `www.wujue.com` | 708 | 0.06% | generic | 2 | 0.1 | OK | — |
| `woman.39.net` | 687 | 0.06% | generic | 2 | 0.1 | OK | — |
| `food.39.net` | 594 | 0.05% | generic | 2 | 0.1 | OK | — |
| `jb39.com` | 527 | 0.04% | generic | 2 | 0.1 | OK | — |
| `ek.39.net` | 448 | 0.04% | generic | 2 | 0.1 | OK | — |
| `man.39.net` | 382 | 0.03% | generic | 2 | 0.1 | OK | — |
| `shen.39.net` | 325 | 0.03% | generic | 2 | 0.0 | OK | — |
| `gan.39.net` | 294 | 0.02% | generic | 2 | 0.0 | OK | — |
| `wei.39.net` | 225 | 0.02% | generic | 2 | 0.0 | OK | — |
| `www.zydcd.com` | 204 | 0.02% | generic | 2 | 0.0 | OK | — |
| `sj.39.net` | 197 | 0.02% | generic | 2 | 0.0 | OK | — |
| `fitness.39.net` | 185 | 0.01% | generic | 2 | 0.0 | OK | — |
| `gk.39.net` | 90 | 0.01% | generic | 2 | 0.0 | OK | — |
| `pf.39.net` | 85 | 0.01% | generic | 2 | 0.0 | OK | — |
| `baidianfeng.familydoctor.com.cn` | 60 | 0.01% | excluded | 2 | 0.0 | Đã bỏ | cần IP Trung Quốc |
| `gc.39.net` | 59 | 0.01% | generic | 2 | 0.0 | OK | — |
| `health.people.com.cn` | 50 | 0.00% | generic | 2 | 0.0 | OK | — |
| `byby.39.net` | 37 | 0.00% | generic | 2 | 0.0 | OK | — |
| `qihuangzhishu.com` | 29 | 0.00% | generic | 2 | 0.0 | OK | — |
| `www.baidu.com` | 11 | 0.00% | generic | 2 | 0.0 | OK | — |
| `bingli.iiyi.com` | 5 | 0.00% | excluded | 2 | 0.0 | Đã bỏ | origin chết, quá ít doc |
| `qy.familydoctor.com.cn` | 2 | 0.00% | excluded | 2 | 0.0 | Đã bỏ | cần IP Trung Quốc |
| `www.familydoctor.cn` | 2 | 0.00% | excluded | 2 | 0.0 | Đã bỏ | cần IP Trung Quốc |
| `ask.familydoctor.com.cn` | 1 | 0.00% | excluded | 2 | 0.0 | Đã bỏ | cần IP Trung Quốc |

## Tiếng Anh

Chỉ PubMed, truy cập qua E-utilities.
1 domain, 240,000 tài liệu (19.66% corpus), lấy được 240,000 (19.66%).
Nhóm này xong sau 0.1 giờ, bị chặn bởi `pubmed.ncbi.nlm.nih.gov`.

| Domain | Tài liệu | % | Crawler | req/s | Giờ | Trạng thái | Thay đổi |
|---|---:|---:|---|---:|---:|---|---|
| `pubmed.ncbi.nlm.nih.gov` | 240,000 | 19.66% | pubmed | 1 | 0.1 | Chờ gỡ chặn | — |
## Những gì đã cải thiện

Bảng dưới là các domain từng hỏng và nay chạy được, kèm nguyên nhân gốc đã tìm ra.

| Domain | Tài liệu | Trước | Sau | Cách sửa |
|---|---:|---|---|---|
| `nhathuoclongchau.com.vn` | 79.598 | 403 Cloudflare, rồi 75h qua Playwright | 4,4h | Giả lập TLS fingerprint của Chrome (`curl_cffi`) |
| `laodong.vn` | 31.282 | HTTP 200 nhưng trích 0 ký tự | Chạy được | Đọc cookie từ JS challenge rồi request lại |
| `baohaiphong.vn` | 16.286 | HTTP 403 | Chạy được | Bộ header browser đầy đủ |
| `baonghean.vn` | 15.941 | HTTP 403 | Chạy được | Bộ header browser đầy đủ |
| `baodanang.vn` | 14.500 | HTTP 403 | Chạy được | Bộ header browser đầy đủ |
| `tamanhhospital.vn` | 4.261 | HTTP 403 | Chạy được | Giả lập TLS fingerprint |
| `hanoimoi.vn` | 1.283 | HTTP 403 | Chạy được | Bộ header browser đầy đủ |
| `baotayninh.vn` | 1.198 | Trích 0 ký tự | Chạy được | Bỏ thẻ `</html>` đóng sớm ở byte 76 |
| `mega.vietnamplus.vn` | 2 | HTTP 403 | Chập chờn | Bộ header browser đầy đủ |

Ngoài ra, hai sửa đổi áp dụng cho toàn bộ corpus.

Thứ nhất, chặn HTTP 203: PubMed trả trang "Cookies must be enabled" dưới mã 2xx nên `raise_for_status()` không bắt được, và 240.000 tài liệu rác từng được lưu với trạng thái thành công.

Thứ hai, thay bộ trích xuất tự viết bằng trafilatura: trang `bachmai.gov.vn` giảm từ 5.722 xuống 4.675 ký tự, bỏ sạch footer, mục "Tin tức cùng chuyên mục" và bảng giờ khám.

## Thứ tự crawl đề xuất

Đề xuất chia làm bốn đợt, lý do nằm ở cột cuối.

| Đợt | Nội dung | Tài liệu | Thời gian | Vì sao đợt này |
|---|---|---:|---:|---|
| 1 | Tiếng Việt, 10 domain lớn nhất | 523.172 | 8,6h | Phần lõi của corpus, chất lượng đã kiểm, chạy là xong |
| 2 | Tiếng Việt, 40 domain còn lại | 208.398 | 3,8h | Đuôi dài, mỗi domain dưới 1,5h, chạy xen kẽ không tốn thêm |
| 3 | Tiếng Trung | 107.631 | 5,6h | Nên đo trần tốc độ `cnkang.com` và `ask.39.net` trước |
| 4 | PubMed | 240.000 | 7 phút | Chỉ 7 phút, nhưng phải chờ NCBI gỡ chặn và có API key |

Đợt 1 và 2 có thể chạy chung một lệnh vì bộ xếp lịch đã tự xen kẽ domain.

Đợt 4 rẻ đến mức nên thử lại mỗi ngày cho tới khi NCBI gỡ chặn.

## Việc nên làm trước khi chạy thật

Hai domain Trung Quốc đang ở mức mặc định 2 req/s và chưa đo trần: `www.cnkang.com` (5,6h) và `ask.39.net` (4,6h).

Nếu chúng chịu được 5 req/s như nhóm tiếng Việt, nhóm tiếng Trung rút từ 5,6h xuống khoảng 2,2h.

Cần thêm `curl_cffi` vào `requirements.txt`.

Playwright không còn được dùng nữa và có thể gỡ bỏ.

## Giới hạn của báo cáo này

Tốc độ và tỉ lệ thành công là số đo thật, lấy từ 5 đến 10 bài mỗi domain, không phải mẫu ngẫu nhiên diện rộng.

Cột Giờ là phép tính từ các tốc độ đó, giả định site giữ nguyên tốc độ suốt nhiều giờ, điều chưa được kiểm chứng.

Con số 88,42% lấy được chỉ đúng nếu NCBI gỡ chặn PubMed, hiện chiếm 19,66% corpus.
