# 2.1 Data Ingestion (Pre-Retrieval Reality)

> RAG quality is bounded by data quality. Clean data in → good retrieval out. Noisy data in → confidently wrong answers out.

## 📌 Key Lesson

You do not need to build parser internals from scratch. Using tools like LlamaParse, Unstructured, or LangExtract is fine — but you must understand **when extraction quality breaks retrieval**.

## Learning Order

The files are ordered to follow the data's journey through the ingestion pipeline:

```
Raw Document → Extract (parser/OCR) → Normalize (canonicalize)
→ Deduplicate → Tag (metadata) → Track (provenance) → Version
```

## Files

| #   | File                                                                                 | Topic                           | Key Concepts                                                                      |
| --- | ------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------------------------------- |
| 01  | [01_clean_vs_noisy_data.md](./01_clean_vs_noisy_data.md)                             | Clean vs Noisy Data             | Data cleaning pipeline, encoding fixes, domain-specific cleaners, quality metrics |
| 02  | [02_parser_ocr_failures.md](./02_parser_ocr_failures.md)                             | Parser & OCR Failures           | Reading-order breaks, missing tables, scanned document extraction                 |
| 03  | [03_document_canonicalization.md](./03_document_canonicalization.md)                 | Document Canonicalization       | Normalized formatting, stable IDs, section path normalization                     |
| 04  | [04_deduplication_strategies.md](./04_deduplication_strategies.md)                   | Deduplication                   | Exact (hash), near-duplicate (MinHash/LSH), semantic dedup, cross-doc dedup       |
| 05  | [05_metadata_design.md](./05_metadata_design.md)                                     | Metadata Design                 | Schema design, chunk-level provenance, stable IDs, section path normalization     |
| 06  | [06_source_attribution_and_provenance.md](./06_source_attribution_and_provenance.md) | Source Attribution & Provenance | Chunk-level provenance tracking, attribution chains, audit trails                 |
| 07  | [07_versioned_documents.md](./07_versioned_documents.md)                             | Versioned Documents             | Version tracking, diff-based updates, temporal metadata, superseded handling      |

## Popular Libraries

| Library       | Purpose                            | Notes                                  |
| ------------- | ---------------------------------- | -------------------------------------- |
| Unstructured  | Multi-format document parsing      | PDFs, HTML, DOCX, images               |
| LlamaParse    | LLM-powered document parsing       | Best for complex layouts               |
| PyMuPDF       | Fast PDF text extraction           | Lightweight, good for text-native PDFs |
| pdfplumber    | PDF table extraction               | Better table handling than PyMuPDF     |
| BeautifulSoup | HTML parsing                       | Standard for web scraping cleanup      |
| datasketch    | MinHash/LSH for near-deduplication | Scalable fuzzy matching                |

## Common Questions

### Q: Which parser should I start with?

**A:** Start with **PyMuPDF** for text-native PDFs and **Unstructured** for mixed formats. If you have complex layouts (scanned docs, tables), try **LlamaParse**. Don't spend days on parsers — pick one and see if extraction quality is good enough for your retrieval.

### Q: Do I need to do ALL these steps (clean, dedup, metadata...)?

**A:** For a prototype, you can skip dedup, versioning, and provenance. The minimum for a working system is: **extract → clean → chunk → embed**. Add the other steps when you need production quality.

### Q: Should metadata be designed before or after chunking?

**A:** Design metadata schema before chunking, because chunking will USE that schema. Each chunk inherits document-level metadata and adds chunk-level metadata (position, section heading, etc.). See the metadata design file for the full schema.

### Q: How does data ingestion connect to the rest of the RAG pipeline?

**A:** Ingestion is the **offline** pipeline. Its output is clean, chunked, metadata-tagged documents ready for embedding. The quality ceiling of your entire RAG system is set here — if important info is lost during parsing or cleaning, no retrieval trick will get it back.

## Syllabus Mapping

Maps to **§2.1** in `p2_rag_depth.md` — covers all checklist items including deduplication strategies (exact, near-duplicate, semantic), versioned documents, metadata design, source attribution, document canonicalization, chunk-level provenance, and parser/OCR failure awareness.
