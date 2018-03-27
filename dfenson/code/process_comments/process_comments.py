
# Import general modules
import sys,os,pandas as pd,math


def process_comments(inpath, infile, outpath):
	fullfile=os.path.join(inpath,infile)
	n_comments=69
	with open(fullfile, 'r', errors='ignore') as f:
		comment=""
		while True:
			comment=f.readline()
			if "COMMENT 1 OF %s" % n_comments in comment:
				break
		for c in range(1, n_comments+1):
			while True:
				comment=f.readline()
				if "Text of Comment:" in comment:
					break
			cout=open(os.path.join(outpath, "comment_%s.txt" % c), 'w')
			while True:
				comment=f.readline()
				if (c < n_comments and "COMMENT %s OF %s" % (str(c+1), n_comments) in comment) or (c==n_comments and comment==""):
						break
				else:
					if not "MACRA Episode-Based Cost Measures Public Comment Summary Report: Verbatim Comments" in comment:
						cout.write(comment)
			cout.close()

def eg_comment_xwalk(inpath, infile, outpath):
	fullfile=os.path.join(inpath, efile)
	df=pd.read_csv(fullfile)
	df=df.loc[:,["EG_DEC", "Comment_No"]]
	df.sort_values("Comment_No", inplace=True)
	egs=pd.DataFrame({'EG_DEC':df.EG_DEC.unique()})
	egs.index.name="eg_id"
	egs["eg_id"]=egs.index
	df=df.merge(egs,'left',on='EG_DEC')
	df.drop('EG_DEC',axis=1, inplace=True)
	result=df.groupby(["Comment_No", "eg_id"]).agg({'eg_id':'first'})
	result=result.unstack()
	result.columns=["eg_%s" % i for i in egs.index]
	result=result.applymap(lambda x: 0 if math.isnan(x) else 1)
	result.to_csv(os.path.join(outpath,"comment_eg_xwalk.csv"))
	egs.drop('eg_id',axis=1,inplace=True)
	egs.to_csv(os.path.join(outpath,"eg.csv"))


if __name__=="__main__":
	inpath="/Users/derek/public_comments/Public_Comments_Tool/Common"
	cfile="verbatim-comments-report.txt"
	efile="comment_excerpts.csv"
	outpath="/Users/derek/public_comments/Public_Comments_Tool/Common"
	process_comments(inpath, cfile, outpath)
	eg_comment_xwalk(inpath, efile, outpath)