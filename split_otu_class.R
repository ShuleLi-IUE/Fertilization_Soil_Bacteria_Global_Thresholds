otu = read.table("./exports/feature-table_w_tax.txt", header=T, sep="\t", row.names=1, comment.char="")

library(dplyr)

dir.create("relative_abundance")
dir.create("absolute_abundance")
# ------------------------
# 门（phylum 4）、纲（class 6）、目（order 8）、科（family 10）、属（genus 12）、种（species 14）
classification_list = c("phylum", "class", "order", "family", "genus", "species")
loc_list = c(4, 6, 8, 10, 12, 14)
# -----------------------------------

for (i in 1:5) {
  otu_rare = otu
  phylum = c()
  delete = c()
  classification = classification_list[i]
  loc = loc_list[i]
  print(paste(classification, "begins"))
  
  count = 1
  for (i in otu_rare[, length(otu_rare)]) 
  {
    tmp = unlist(strsplit(as.character(i), split="; |__"))
    phylum = c(phylum, tmp[loc])
    
    if (is.na(tmp[loc]) | tmp[loc] == "")
    {
      delete = c(delete, count)
    }
    count = count + 1
  }
  
  delete
  otu_rare$phylum = phylum
  otu_rare  = otu_rare[, -(ncol(otu_rare)-1)] # delete annotation
  otu_rare2 = otu_rare[-delete, ] # delete NA and ""
  
  otu_rare2 = otu_rare2[order(otu_rare2$phylum), ]
  plist = unique(otu_rare2$phylum)
  plist[1]
  
  c(1:(ncol(otu_rare2)-1))
  test = otu_rare2[otu_rare2$phylum == plist[1], c(1:(ncol(otu_rare2)-1))]
  
  # merge
  rare3 = data.frame(apply(otu_rare2[otu_rare2$phylum == plist[1], c(1:(ncol(otu_rare2)-1))], 2, sum))
  colnames(rare3)[1] = plist[1]
  for (i in 2:length(plist))
  {
    tmp = apply(otu_rare2[otu_rare2$phylum == plist[i], c(1:(ncol(otu_rare2)-1))], 2, sum)
    rare3 = cbind(rare3, tmp)
    colnames(rare3)[i] = plist[i]
  }
  rare3 = data.frame(t(rare3))
  filename = paste("./absolute_abundance/", classification, "_otu_absolute_abun.csv",sep="")
  write.csv(rare3, file = filename)
  
  # normalize
  norm = rare3
  sample_sum = apply(rare3, 2, sum)
  
  for (i in 1:nrow(rare3))
  {
    for (j in 1:ncol(rare3))
    {
      norm[i, j] = rare3[i, j]/sample_sum[j]
    }
  }
  apply(norm, 2, sum) # validation
  filename_relativate = paste("./relative_abundance/", classification, "_otu_relative_abun.csv",sep="")
  write.csv(norm, file = filename_relativate)
}









