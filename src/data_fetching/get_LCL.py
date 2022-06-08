import shutil
import urllib.request as request
from contextlib import closing
from pathlib import Path
import pandas as pd
import argparse

                
def get_LCL_fastq(file_dir="."):
    meta = pd.read_csv((Path(__file__) / Path("../PRJEB3366_tsv.txt")), sep='\t')
    for i, row in meta.iterrows():
        for fq in row["fastq_ftp"].split(";"):
            file_name = Path(fq).name
            fq = f"ftp://{fq}"
            print(f"Downloading {file_name} from {fq}")
            
            file_dir = Path(file_dir)
            
            if (file_dir / file_name).is_file():
                print(file_name, " file is exist")
                continue

            with closing(request.urlopen(fq)) as r:
                with open(file_dir / file_name, 'wb') as f:
                    shutil.copyfileobj(r, f)
                    print(file_name, " file is downloaded")
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='get_LCL.py', description='get LCL fastq file from the database') 
    parser.add_argument('--output', '-o', default='.', type=str, required=False, help='Folder to save the files')
    args = parser.parse_args()

    get_LCL_fastq(args.output)