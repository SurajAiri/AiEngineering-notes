# 2.1 Data Ingestion (Pre-Retrieval Reality)

> RAG quality is bounded by data quality. Clean data in → good retrieval out. Noisy data in → confidently wrong answers out.

## 📌 Key Lesson

You do not need to build parser internals from scratch. Using tools like LlamaParse, Unstructured, or LangExtract is fine — but you must understand **when extraction quality breaks retrieval**.

## Files

| File                                                                                 | Topic                           | Key Concepts                                                                      |
| ------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------------------------------- |
| [01_clean_vs_noisy_data.md](./01_clean_vs_noisy_data.md)                             | Clean vs Noisy Data             | Data cleaning pipeline, encoding fixes, domain-specific cleaners, quality metrics |
| [02_deduplication_strategies.md](./02_deduplication_strategies.md)                   | Deduplication                   | Exact (hash), near-duplicate (MinHash/LSH), semantic dedup, cross-doc dedup       |
| [03_versioned_documents.md](./03_versioned_documents.md)                             | Versioned Documents             | Version tracking, diff-based updates, temporal metadata, superseded handling      |
| [04_metadata_design.md](./04_metadata_design.md)                                     | Metadata Design                 | Schema design, chunk-level provenance, stable IDs, section path normalization     |
| [05_source_attribution_and_provenance.md](./05_source_attribution_and_provenance.md) | Source Attribution & Provenance | Chunk-level provenance tracking, attribution chains, audit trails                 |
| [06_document_canonicalization.md](./06_document_canonicalization.md)                 | Document Canonicalization       | Normalized formatting, stable IDs, section path normalization                     |
| [07_parser_ocr_failures.md](./07_parser_ocr_failures.md)                             | Parser & OCR Failures           | Reading-order breaks, missing tables, scanned document extraction                 |

## Syllabus Mapping

Maps to **§2.1** in `p2_rag_depth.md` — covers all checklist items including deduplication strategies (exact, near-duplicate, semantic), versioned documents, metadata design, source attribution, document canonicalization, chunk-level provenance, and parser/OCR failure awareness.
