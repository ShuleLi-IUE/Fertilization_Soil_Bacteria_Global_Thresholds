otu = read.table("./exports/feature-table_w_tax.txt", header=T, sep="\t", row.names=1, comment.char="")
result_path <- "YOURPATH"

library(vegan)
spe <- otu
spe <- spe[,-ncol(spe)]
spe <- t(spe)

shannon<-diversity(spe,index = "shannon") # slow
simpson<-diversity(spe,index = "invsimpson")
chao1<-estimateR(spe)[2,]
ACE<-estimateR(spe)[4,]
richness<-estimateR(spe)[1,]
# Pielou evenness
pielou_evenness <- shannon/ log(specnumber(spe), exp(1))
gini_simpson_index <- diversity(spe, index = 'simpson')
# equitability evenness
equitability <- 1 / (richness * (1 - gini_simpson_index))
alfa<-data.frame(Richness=richness,Shannon=shannon,InvSimpson=simpson,Chao1=chao1,ACE=ACE,
                 Pielou_Evenness=pielou_evenness, Equitability_Evenness=equitability)

write.csv(alfa, file = "OUTPUT_PATH")
