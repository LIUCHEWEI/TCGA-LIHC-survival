# Load required packages
library(maftools)  # For handling MAF files
library(dplyr)     # For data manipulation
library(tidyr)     # For data reshaping
library(ggplot2)   # For visualization
library(GenomicRanges)  # For genomic interval operations
library(facets)
# Read MAF file
lihc_maf <- read.maf("TCGA_LIHC_only.maf")

# Extract relevant mutation information
mutations <- as.data.frame(lihc_maf@data) %>%
  select(Tumor_Sample_Barcode, Hugo_Symbol, Chromosome, Start_Position, End_Position, 
         Reference_Allele, Tumor_Seq_Allele2, Variant_Type, Variant_Classification,
         t_ref_count, t_alt_count, n_ref_count, n_alt_count) %>%
  # Calculate variant allele fraction (VAF)
  mutate(
    tumor_depth = t_ref_count + t_alt_count,
    normal_depth = n_ref_count + n_alt_count,
    tumor_vaf = t_alt_count / tumor_depth
  ) %>%
  # Remove non-exonic or silent mutations 
  filter(!(Variant_Classification %in% c("Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "IGR")))

# Create a unique identifier for each mutation
mutations$mutation_id <- paste(mutations$Chromosome, mutations$Start_Position, 
                               mutations$Reference_Allele, mutations$Tumor_Seq_Allele2, sep="_")

# Convert Chromosome to the same format as in copy number data 
mutations$Chromosome <- gsub("chr", "", mutations$Chromosome)

cat("Processed", nrow(mutations), "mutations from", length(unique(mutations$Tumor_Sample_Barcode)), "samples\n")


# Directory containing the snp-pileup files
snp_pileup_dir <- "~/Desktop/output"  
snp_pileup_files <- list.files(snp_pileup_dir, pattern = "*.snp_pileup.csv.gz$", full.names = TRUE)

# Process all samples and store copy number segments
all_segments <- list()
sample_purity <- list()

for (file_path in snp_pileup_files) {
  # Extract sample ID from filename
  sample_id <- gsub(".*TCGA-(..-....).*", "TCGA-\\1", basename(file_path))
  
  cat("Processing", sample_id, "from file", basename(file_path), "\n")
  
  tryCatch({
    # Read the snp-pileup file
    # Note: The column structure is: Chromosome, Position, Ref, Alt, File1R, File1A, File1E, File1D, File2R, File2A, File2E, File2D
    snp_matrix <- readSnpMatrix(file_path)
    
    # Pre-process the data
    xx <- preProcSample(snp_matrix, cval = 225)
    
    # Run FACETS segmentation
    oo <- procSample(xx, cval = 225)
    
    # Fit copy number model
    fit <- emcncf(oo)
    
    # Extract segments with copy number information
    segments <- fit$cncf
    
    # Add sample ID to segments
    segments$sample_id <- sample_id
    
    # Store segments
    all_segments[[length(all_segments) + 1]] <- segments
    
    # Store purity information
    sample_purity[[length(sample_purity) + 1]] <- data.frame(
      sample_id = sample_id,
      purity = fit$purity,
      ploidy = fit$ploidy,
      dipLogR = oo$dipLogR
    )
    
    cat("Successfully processed", sample_id, 
        "- Purity:", round(fit$purity, 3), 
        "Ploidy:", round(fit$ploidy, 3), "\n")
    
  }, error = function(e) {
    cat("Error processing", sample_id, ":", conditionMessage(e), "\n")
  })
}

# Combine all segments into a single data frame
if (length(all_segments) > 0) {
  copy_number <- bind_rows(all_segments)
  sample_purity_df <- bind_rows(sample_purity)
  
  cat("Processed copy number data for", length(all_segments), "samples\n")
  
  # Standardize column names for subsequent steps
  copy_number <- copy_number %>%
    select(
      chrom = chrom,
      start = start,
      end = end,
      tcn = tcn.em,       # Total copy number from EM algorithm
      lcn = lcn.em,       # Minor allele copy number from EM algorithm
      cf = cf.em,         # Cellular fraction from EM algorithm
      sample_id = sample_id
    ) %>%
    mutate(
      mcn = tcn - lcn     # Calculate major copy number
    )
} else {
  stop("No FACETS segments could be generated")
}

# Save the copy number data for future use
write.csv(copy_number, "lihc_copy_number_segments225.csv", row.names = FALSE)
write.csv(sample_purity_df, "lihc_sample_purity225.csv", row.names = FALSE)

get_copy_number_state <- function(mutations, copy_number) {
  # 將樣本ID統一為TCGA-XX-XXXX格式
  mutations$standard_id <- gsub("^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}).*", "\\1", mutations$Tumor_Sample_Barcode)
  
  # 將染色體統一為字符串格式，並確保沒有前綴
  mutations$Chromosome <- as.character(gsub("^chr", "", mutations$Chromosome))
  copy_number$chrom <- as.character(copy_number$chrom)  # 確保拷貝數檔案的染色體為字符串
  
  # 顯示處理前的信息
  cat("將處理", length(unique(mutations$standard_id)), "個樣本的突變資料\n")
  cat("總共有", nrow(mutations), "個突變需要匹配\n")
  
  # 轉換為GRanges objects
  mutation_gr <- GRanges(
    seqnames = mutations$Chromosome,
    ranges = IRanges(start = mutations$Start_Position, end = mutations$End_Position),
    mutation_id = mutations$mutation_id,
    standard_id = mutations$standard_id
  )
  
  # 處理每個樣本
  results <- list()
  processed_samples <- 0
  matched_mutations <- 0
  
  for (sample_id in unique(mutations$standard_id)) {
    # 選取該樣本的突變
    sample_mutations <- mutation_gr[mutation_gr$standard_id == sample_id]
    
    # 選取該樣本的拷貝數資料
    sample_cn <- copy_number[copy_number$sample_id == sample_id, ]
    
    if (nrow(sample_cn) == 0) {
      cat("警告：找不到樣本", sample_id, "的拷貝數資料\n")
      next
    }
    
    processed_samples <- processed_samples + 1
    
    # 轉換拷貝數資料為GRanges objects
    cn_gr <- GRanges(
      seqnames = sample_cn$chrom,
      ranges = IRanges(start = sample_cn$start, end = sample_cn$end),
      tcn = sample_cn$tcn,
      lcn = sample_cn$lcn,
      mcn = sample_cn$mcn,
      cf = sample_cn$cf
    )
    
    # 找出突變與拷貝數區段的重疊
    overlaps <- findOverlaps(sample_mutations, cn_gr)
    
    if (length(overlaps) > 0) {
      # 提取每個突變的拷貝數資訊
      mutation_cn <- data.frame(
        mutation_id = sample_mutations$mutation_id[queryHits(overlaps)],
        sample_id = sample_id,
        tcn = cn_gr$tcn[subjectHits(overlaps)],
        lcn = cn_gr$lcn[subjectHits(overlaps)],
        mcn = cn_gr$mcn[subjectHits(overlaps)],
        cf = cn_gr$cf[subjectHits(overlaps)]
      )
      
      matched_mutations <- matched_mutations + nrow(mutation_cn)
      results[[length(results) + 1]] <- mutation_cn
    }
    
    # 顯示進度（每10個樣本）
    if (processed_samples %% 10 == 0) {
      cat("已處理", processed_samples, "個樣本，當前匹配了", matched_mutations, "個突變\n")
    }
  }
  
  # 顯示最終結果
  cat("處理完成。共處理了", processed_samples, "個樣本，成功匹配了", matched_mutations, "個突變\n")
  
  # 合併所有樣本的結果
  if (length(results) > 0) {
    combined_results <- bind_rows(results)
    return(combined_results)
  } else {
    stop("沒有突變可以與拷貝數資料匹配")
  }
}

# 使用修改後的函數
mutation_cn <- get_copy_number_state(mutations, copy_number)

# 合併拷貝數資訊回突變資料框架
mutations_with_cn <- mutations %>%
  # 添加標準化樣本ID
  mutate(standard_id = gsub("^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}).*", "\\1", Tumor_Sample_Barcode)) %>%
  # 使用標準化ID和突變ID合併
  left_join(mutation_cn, by = c("mutation_id", "standard_id" = "sample_id"))

cat("成功為", sum(!is.na(mutations_with_cn$tcn)), "個突變分配拷貝數狀態\n")
cat("這佔總突變數的", round(sum(!is.na(mutations_with_cn$tcn)) / nrow(mutations_with_cn) * 100, 2), "%\n")

# Function to estimate mutation multiplicity based on VAF, purity, and copy number
estimate_multiplicity_advanced <- function(alt_count, total_count, purity, tcn, lcn, normal_cn = 2, 
                                           pre_calculated_vaf = NULL, conf_level = 0.95) {
  # 檢查輸入參數是否有 NA 或無限值
  if (is.na(alt_count) || is.na(total_count) || is.na(purity) || is.na(tcn) || 
      !is.finite(alt_count) || !is.finite(total_count) || !is.finite(purity) || !is.finite(tcn) ||
      total_count == 0 || purity == 0) {
    return(list(
      multiplicity = 1,  # 預設多重性
      cellular_fraction = NA,
      is_clonal = NA,
      mc_lower = NA,
      mc_upper = NA,
      vaf_observed = NA
    ))
  }
  
  # 使用預先計算的 VAF，否則重新計算
  vaf_observed <- if(!is.null(pre_calculated_vaf) && !is.na(pre_calculated_vaf)) {
    pre_calculated_vaf
  } else {
    alt_count / total_count
  }
  
  # 檢查VAF和計算條件
  if (vaf_observed <= 0 || purity <= 0) {
    return(list(
      multiplicity = 1,
      cellular_fraction = 0,
      is_clonal = FALSE,
      mc_lower = 0,
      mc_upper = 0,
      vaf_observed = vaf_observed
    ))
  }
  
  # 計算 VAF 的信賴區間 (仍然使用原始計數，因為二項分布檢定需要原始數據)
  ci <- tryCatch({
    binom.test(alt_count, total_count, conf.level = conf_level)$conf.int
  }, error = function(e) {
    c(vaf_observed, vaf_observed)  # 在錯誤時使用觀察到的 VAF 作為置信區間
  })
  
  vaf_lower <- ci[1]
  vaf_upper <- ci[2]
  
  # 計算 mC 的信賴區間
  denominator <- purity * tcn + (1 - purity) * normal_cn
  
  mc_lower <- vaf_lower * denominator / purity
  mc_upper <- vaf_upper * denominator / purity
  
  # 檢查 mc_lower 和 mc_upper 是否為有限數字
  if (!is.finite(mc_lower) || !is.finite(mc_upper)) {
    return(list(
      multiplicity = min(round(vaf_observed * denominator / purity), tcn),
      cellular_fraction = 1,
      is_clonal = TRUE,
      mc_lower = NA,
      mc_upper = NA,
      vaf_observed = vaf_observed
    ))
  }
  
  # 根據論文規則確定多重性和細胞分數
  # 檢查 mc_lower 和 mc_upper 之間是否存在整數
  integers_in_range <- integer(0)
  if (mc_lower <= mc_upper) {  # 確保區間有效
    lower_int <- ceiling(mc_lower)
    upper_int <- floor(mc_upper)
    if (lower_int <= upper_int) {
      integers_in_range <- lower_int:upper_int
    }
  }
  
  if (length(integers_in_range) > 0) {
    # 包含整數，克隆性突變
    m <- integers_in_range[1]  # 取第一個整數
    c <- 1.0
    is_clonal <- TRUE
  } else if (mc_upper < 1) {
    # 整個區間低於1
    m <- 1
    c <- mc_upper
    is_clonal <- c > 0.75
    if (is_clonal) c <- 1.0
  } else {
    # 整個區間高於1且不包含整數
    m <- 2  # 預設起始值
    c <- 1.0
    is_clonal <- TRUE
    
    # 尋找合適的多重性值
    for (possible_m in 2:tcn) {
      possible_c_lower <- mc_lower / possible_m
      possible_c_upper <- mc_upper / possible_m
      
      if (possible_c_lower <= 1 && possible_c_upper >= 0) {
        m <- possible_m
        c <- min(1.0, possible_c_upper)
        is_clonal <- c > 0.75
        break
      }
    }
  }
  
  # 確保 m 不大於總拷貝數且至少為1
  m <- min(max(1, m), tcn)
  
  return(list(
    multiplicity = m,
    cellular_fraction = c,
    is_clonal = is_clonal,
    mc_lower = mc_lower,
    mc_upper = mc_upper,
    vaf_observed = vaf_observed
  ))
}

# 過濾位於性染色體上的突變
mutations_with_cn <- mutations_with_cn %>%
  filter(!(Chromosome %in% c("X", "Y", "chrX", "chrY")))

mutations_classified <- mutations_with_cn %>%
  rowwise() %>%
  mutate(
    # 使用進階函數估計多重性，並傳入預先計算的 tumor_vaf
    mult_result = list(estimate_multiplicity_advanced(
      alt_count = ifelse(is.na(t_alt_count), 0, t_alt_count),
      total_count = ifelse(is.na(tumor_depth), 0, tumor_depth),
      purity = ifelse(is.na(cf), 0.5, cf),  # 使用默認純度0.5如果NA
      tcn = ifelse(is.na(tcn), 2, tcn),     # 使用默認拷貝數2如果NA
      lcn = ifelse(is.na(lcn), 1, lcn),     # 使用默認低拷貝數1如果NA
      pre_calculated_vaf = tumor_vaf,       # 使用預先計算的 VAF
      normal_cn = 2
    )),
    
    # 從結果中提取值
    multiplicity = mult_result$multiplicity,
    cellular_fraction = mult_result$cellular_fraction,
    is_clonal = mult_result$is_clonal,
    
    # 分類突變
    is_multi_copy = !is.na(multiplicity) & multiplicity > 1,
    is_only_copy = !is.na(tcn) & tcn == 1,
    is_persistent = is_multi_copy | is_only_copy,
    
    # 突變類別
    mutation_category = case_when(
      is_multi_copy ~ "Multi-copy",
      is_only_copy ~ "Only-copy",
      TRUE ~ "Loss-prone"
    )
  ) %>%
  ungroup()

# Summarize classification results
classification_summary <- mutations_classified %>%
  group_by(mutation_category) %>%
  summarize(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

print(classification_summary)

# 計算每個樣本的pTMB
ptmb_per_sample <- mutations_classified %>%
  group_by(standard_id) %>%  # 使用標準化樣本ID
  summarize(
    total_mutations = n(),
    multi_copy_mutations = sum(is_multi_copy, na.rm = TRUE),
    only_copy_mutations = sum(is_only_copy, na.rm = TRUE),
    persistent_mutations = sum(is_persistent, na.rm = TRUE),  # persistent = multi_copy + only_copy
    loss_prone_mutations = sum(!is_persistent, na.rm = TRUE),
    pTMB = persistent_mutations
  ) %>%
  # 計算各類突變佔比
  mutate(
    multi_copy_percent = multi_copy_mutations / total_mutations * 100,
    only_copy_percent = only_copy_mutations / total_mutations * 100,
    persistent_percent = persistent_mutations / total_mutations * 100,
    loss_prone_percent = loss_prone_mutations / total_mutations * 100
  )

# 顯示基本統計資訊
cat("===== LIHC樣本pTMB統計摘要 =====\n")
cat("總樣本數:", nrow(ptmb_per_sample), "\n")
cat("平均pTMB:", mean(ptmb_per_sample$pTMB), "\n")
cat("中位數pTMB:", median(ptmb_per_sample$pTMB), "\n")
cat("最小pTMB:", min(ptmb_per_sample$pTMB), "\n")
cat("最大pTMB:", max(ptmb_per_sample$pTMB), "\n")

# 輸出前10個樣本的pTMB值
cat("\n前10個樣本的pTMB值:\n")
print(head(ptmb_per_sample, 10))

# 將結果儲存為CSV檔案
write.csv(ptmb_per_sample, "pTMB_results_225.csv",  row.names = FALSE)
