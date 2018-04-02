
# Import general modules
import sys,os,pandas as pd,math

TEXT_TO_REMOVE=["MACRA Episode-Based Cost Measures Public Comment Summary Report: Verbatim Comments"]

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
					if sum([text in comment for text in TEXT_TO_REMOVE])==0:
						cout.write(comment)
			cout.close()

def process_eg_labels(inpath, infile, outpath):
	fullfile=os.path.join(inpath, efile)
	df=pd.read_csv(fullfile)
	df=df.loc[df.Comment_Period=="2016_12_macra_posting",["EG_DEC", "Comment_No", "Component_Description","Episode_Theme_Description"]]
	df.rename(columns={'EG_DEC':"episode_group",
				'Component_Description':'component',
				'Comment_No':'comment_id',
				'Episode_Theme_Description':'theme'}, inplace=True)
	df.sort_values("comment_id", inplace=True)
	df['value']=1
	# Create ids for each potential label, and output description csv files for each
	for col in ["episode_group", "component", "theme"]:
		df["%s_id" % col]=df.groupby(col).grouper.group_info[0]
		result=df.loc[:,[col,"%s_id" % col]].drop_duplicates(col)
		result.set_index("%s_id" % col, inplace=True)
		result.sort_index(inplace=True)
		result.to_csv(os.path.join(outpath,"%s.csv" % col))
		df.drop(col, axis=1, inplace=True)
		result=df.drop_duplicates(["comment_id", "%s_id" % col]).pivot(index='comment_id',columns="%s_id" % col, values='value')
		result=result.applymap(lambda x: 0 if math.isnan(x) else 1)
		result.columns=["{0}_{1}".format(col, c) for c in result.columns]
		result.to_csv(os.path.join(outpath,"%s_labels.csv" % col))


if __name__=="__main__":
	inpath="/Users/derek/public_comments/Public_Comments_Tool/Common"
	cfile="verbatim-comments-report.txt"
	efile="comment_excerpts.csv"
	outpath="/Users/derek/public_comments/Public_Comments_Tool/Common/data"	
	process_comments(inpath, cfile, outpath)
	process_eg_labels(inpath, efile, outpath)

