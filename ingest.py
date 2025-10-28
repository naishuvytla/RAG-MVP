# imports for filesystem paths json file matching http requests and xml parsing
import os, json, glob, requests
from lxml import etree

# url for the grobid endpoint that converts a pdf into tei xml
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# input and output folders
IN_DIR, OUT_DIR = "docs", "tei"

# create the tei folder if it does not exist
os.makedirs(OUT_DIR, exist_ok=True)

def pdf_to_tei(pdf_path):
    # open the pdf in binary mode so the http upload works correctly
    with open(pdf_path, "rb") as f:
        # post the file to grobid and ask it to consolidate citations
        r = requests.post(GROBID_URL, files={"input": f}, data={"consolidateCitations": 1})
    # raise an error if grobid returned a bad status code
    r.raise_for_status()
    # return the tei xml as a string
    return r.text

def tei_to_text(tei_xml):
    # parse the tei xml into an element tree
    root = etree.fromstring(tei_xml.encode("utf-8"))
    # namespace mapping for tei xpath queries
    ns = {"t":"http://www.tei-c.org/ns/1.0"}
    # extract the title text from the tei header
    title = " ".join(root.xpath("//t:titleStmt/t:title//text()", namespaces=ns)) or "Untitled"
    # extract all body text from the tei
    body  = " ".join(root.xpath("//t:text//t:body//text()", namespaces=ns))
    # join title and body and replace newlines with spaces
    text  = (title + "\n" + body).replace("\n", " ")
    # trim and collapse extra whitespace
    return title.strip(), " ".join(text.split())

def make_chunks(text, chunk_chars=1200, overlap=250):
    # slice the text into overlapping windows to improve recall
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_chars])
        # move forward by chunk size minus overlap and guard against zero step
        i += max(1, chunk_chars - overlap)
    return chunks

def main():
    # a list that will hold all chunk objects for all pdfs
    all_chunks = []
    # find all pdf files in the docs folder
    pdfs = glob.glob(os.path.join(IN_DIR, "*.pdf"))
    if not pdfs:
        print("No PDFs found in docs/. Add a PDF and rerun.")
        return
    # process each pdf one by one
    for pdf in pdfs:
        # base filename without extension used for ids and paths
        base = os.path.splitext(os.path.basename(pdf))[0]
        print("Processing:", base)
        # run the pdf through grobid to get tei xml
        tei = pdf_to_tei(pdf)
        # save the raw tei so you can inspect or reuse it later
        open(os.path.join(OUT_DIR, base + ".tei.xml"), "w", encoding="utf-8").write(tei)
        # convert tei into a clean title and plain text
        title, text = tei_to_text(tei)
        # split the text into chunks and store metadata for each chunk
        for j, c in enumerate(make_chunks(text)):
            all_chunks.append({
                "doc": base, "title": title,
                "chunk_id": f"{base}_{j}",
                # prepend the title so each chunk carries document identity
                "text": f"{title}\n{c}"
            })
    # write all chunks for all pdfs to a single json file
    json.dump(all_chunks, open("corpus.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Ingested {len(all_chunks)} chunks from {len(pdfs)} PDFs.")

if __name__ == "__main__":
    # run the ingest pipeline when this file is executed as a script
    main()
