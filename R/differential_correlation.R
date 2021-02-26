library(DGCA)
data = read.csv("./data/432 lipid.csv",header = T,row.names = 1)
data1 = data

n_case = 89
n_control = 178
cell_type = c(rep('case',n_case),rep('control',n_control))
design_mat1 = makeDesign(cell_type)


#使用spearman相关系数，生成包含differential correlation的dataframe
ddcor_res1 = ddcorAll(inputMat = data1, design = design_mat1,compare = c("case", "control"),
                     adjust = "BH", heatmapPlot = FALSE,nPerms = 0,corrType = "spearman", nPairs = 'all')
write.csv(ddcor_res1,'./data/differential corr.CSV')




