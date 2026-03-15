#!/usr/bin/env python3
"""
Convert Markdown writeups to HTML using pandoc.
The HTML files can be printed to PDF from a browser.
"""

import os
import subprocess
from pathlib import Path


def convert_md_to_html(md_path, html_path):
    """Convert a markdown file to styled HTML using pandoc."""
    # Use pandoc with standalone HTML and nice styling
    cmd = [
        'pandoc',
        str(md_path),
        '-o', str(html_path),
        '--standalone',
        '--metadata', 'title=Project Writeup',
        '--css', 'https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown.min.css',
        '--template', '-'
    ]
    
    # Custom HTML template
    template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>$title$</title>
    <style>
        body {
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }
        h1 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        h2 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        code { background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; }
        pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #dfe2e5; padding: 6px 13px; }
        th { background-color: #f6f8fa; }
        img { max-width: 100%; }
        @media print {
            body { max-width: none; padding: 20px; }
        }
    </style>
</head>
<body>
$body$
</body>
</html>"""
    
    # Run pandoc with template from stdin
    result = subprocess.run(
        ['pandoc', str(md_path), '-o', str(html_path), '--standalone', 
         '--template=-', '--metadata', 'title=Project Writeup'],
        input=template,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Created: {html_path}")
    else:
        print(f"Error: {result.stderr}")


def main():
    base_dir = Path("/Users/tilekbekzhoroev/Desktop/UQ-Update")
    
    # Convert each project README to HTML
    for i in range(1, 7):
        project_dir = base_dir / f"Project {i}"
        md_file = project_dir / "README.md"
        html_file = project_dir / f"Project_{i}_writeup.html"
        
        if md_file.exists():
            print(f"Converting Project {i}...")
            convert_md_to_html(str(md_file), str(html_file))
        else:
            print(f"Warning: {md_file} not found")
    
    print("\nHTML files created. To generate PDFs:")
    print("1. Open each HTML file in a browser")
    print("2. Use Print -> Save as PDF")


if __name__ == "__main__":
    main()
