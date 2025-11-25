# Raw Data Directory

Place your raw data files here for the RAG pipeline to process.

## Supported File Types
The pipeline uses `SimpleDirectoryReader` and supports the following formats:
- **PDF** (`.pdf`): Research papers, slides, documentation.
- **Text** (`.txt`): Plain text notes, logs.
- **Markdown** (`.md`): Documentation, notes.
- **Word** (`.docx`): Documents.
- **PowerPoint** (`.pptx`): Presentations.
- **CSV/Excel**: Structured data (may require specific formatting).

## Automated Downloads
The `gather` script will automatically populate this directory with top ArXiv papers on specified topics.
