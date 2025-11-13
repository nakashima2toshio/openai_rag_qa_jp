
Q/Aペア：
qa_output/a02_qa_pairs_cc_news.csv  4,646件
qa_output/a02_qa_pairs_livedoor.csv   965件

### Qdrant-colledtion list

• raw_cc_news (660件)
• qa_cc_news_a02_llm (9,290件)
• qa_cc_news_a03_rule (8,556件)
• qa_cc_news_a10_hybrid (4,334件)

• raw_livedoor (20,193件)
• qa_livedoor_a02_20_llm (964件)
• qa_livedoor_a03_rule (466件)
• qa_livedoor_a10_hybrid (2,175件)

### colledtion list for csv
qa_output/a02_qa_pairs_cc_news.csv
qa_output/a02_qa_pairs_livedoor.csv
qa_output/a03_qa_pairs_cc_news.csv
qa_output/a03_qa_pairs_livedoor.csv
qa_output/a10_qa_pairs_cc_news.csv
qa_output/a10_qa_pairs_livedoor.csv

### %ls -1d *.py

a01_load_non_qa_rag_data.py
a02_make_qa.py
a03_rag_qa_coverage_improved.py
a10_qa_optimized_hybrid_batch.py
a20_output_qa_csv.py
a31_make_cloud_vector_store_vsid.py
a34_rag_search_cloud_vs.py
a40_show_qdrant_data.py
a41_qdrant_truncate.py
a42_qdrant_registration.py
a50_rag_search_local_qdrant.py

coverage_japan.py
helper_api.py
helper_rag.py
helper_rag_qa.py
helper_st.py

qa_keyword_extractor.py
regex_mecab.py
regex_vs_mecab.py
server.py
setup.py

test_mecab_format.py
translate_cc_news.py

