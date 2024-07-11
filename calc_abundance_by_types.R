# ----------change here--------------
project <- "FertiType"
# -----------------------------------
pwd <- "/Users/YourPath"
pwd_ <- "/Users/YourPath/"
setwd(pwd)

library(dplyr)
group_base <- read.csv("./groups/group_information.csv") %>% dplyr::select(sample, Type = FertiType)
group_base$Type %>% table

# level of classification
classifications = c("phylum", "class", "order", "family", "genus", "species")
# class = "family"

for (class in classifications) {
  # 物种组成丰度
  print(paste(class, "begins..."))
  setwd(pwd)
  wd <- paste0(pwd, "/projects/", project, "/", class)
  dir.create(wd, recursive = TRUE)
  setwd(wd)
  
  otu_phylum <- read.csv(paste(pwd_,"relative_abundance/", class, "_otu_relative_abun.csv", sep=""))
  
  
  save_csv = paste("abundance_all_",project, "_", class, ".csv", sep="")
  
  rownames(otu_phylum) <- otu_phylum[, 1]
  otu_phylum <- otu_phylum[, -1]
  otu_phylum_filter <- otu_phylum
  otu_phylum_filter$sum <- rowSums(otu_phylum_filter)
  
  otu_phylum_filter <- otu_phylum_filter[order(otu_phylum_filter$sum, decreasing = TRUE), ]
  phylum_dec <- otu_phylum_filter[, -ncol(otu_phylum_filter)]
  phylum_dec_count <- phylum_dec
  # exclude zeros
  phylum_dec_count$Frequency <- (ncol(phylum_dec) - rowSums(phylum_dec == 0)) / ncol(phylum_dec) 
  
  library(dplyr)
  group <- group_base 
  names(group)[1] <- 'variable'
  # filter group
  group <- filter(group, variable %in% intersect(group$variable, colnames(phylum_dec)))
  phylum_dec_plus <- phylum_dec_count
  
  types <- levels(factor(group$Type))
  # ----------different types--------------
  phylum_dec_plus$CK   <- rowMeans(phylum_dec_plus[, filter(group, Type == "CK")$variable])
  phylum_dec_plus$IF <- rowMeans(phylum_dec_plus[, filter(group, Type == "IF")$variable])
  phylum_dec_plus$IFOF <- rowMeans(phylum_dec_plus[, filter(group, Type == "IFOF")$variable])
  phylum_dec_plus$OF <- rowMeans(phylum_dec_plus[, filter(group, Type == "OF")$variable])
  
  phylum_dec_plus <- select(phylum_dec_plus, Frequency, CK, IF, IFOF, OF)
  
  write.csv(phylum_dec_plus, save_csv)
}



