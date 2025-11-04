# Test Data for File Processor API

This directory contains test files for validating the enhanced file processor API.

## PDF Test Files

The `pdfs/` directory contains sample PDF documents for testing:

- **`top-100-movies.pdf`** (49KB) - Simple movie list, good for basic chunking tests
- **`US-Youth-Soccer-Travel-Policy.pdf`** (187KB) - Multi-page policy document
- **`travel program document.pdf`** (257KB) - Travel program documentation
- **`CommissionAgreement2020+(2).pdf`** (196KB) - Legal agreement document
- **`focs.pdf`** (6.3MB) - Large academic paper
- **`wdf520padm.pdf`** (9.4MB) - Large technical document

## Usage

These files are used by the test suite:

```bash
python test_real_pdfs.py        # Tests multiple PDFs with different chunking strategies
python test_large_pdf.py        # Performance testing with larger documents
```

## Test Coverage

The test files provide coverage for:
- Small documents (< 50KB)
- Medium documents (100-500KB)
- Large documents (> 1MB)
- Various content types (lists, policies, academic papers, technical docs)
- Multi-page documents with complex structure