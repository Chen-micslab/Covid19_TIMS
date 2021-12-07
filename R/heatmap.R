library(pheatmap)
data = read.csv('./data/area matrix.CSV',row.names = 1,header = T,stringsAsFactors = F)
data1 = data
data1 = log10(data1)
for (i in 1:length(row.names(data1))){
  a = as.numeric(data1[i,])
  b = scale(a,center = TRUE,scale = TRUE)
  data1[i,] = b
}
z = read.csv('./data/annotation.CSV',header = T,stringsAsFactors = F)
x=colnames(data) 
rownames(z)<-x
bk = c(seq(-9,-0.1,by=0.02),seq(0,9,by=0.02))
ac = list(
  Type = c(Asymptomatic = "#990000", Healthy = "#663399"),
  Sex = c(Male = "#cc9900", Female = "#336600")
)
pheatmap(data1,scale = 'none',annotation_col=z,cluster_cols = FALSE,show_colnames = F,show_rownames = T,cluster_rows = FALSE,fontsize_row = 10, 
         color = c(colorRampPalette(colors = c("blue","white"))(length(bk)/2),colorRampPalette(colors = c("white","red"))(length(bk)/2)),
         legend_breaks=seq(-8,8,2),
         breaks=bk,cellwidth = 2,
         cellheight = 10,
         annotation_colors = ac,
         filename = './data/Heatmap.pdf')


  
