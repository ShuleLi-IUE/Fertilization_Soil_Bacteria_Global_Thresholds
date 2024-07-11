library(dplyr)
df_models_g <- df_models %>% group_by(Model, x) %>% 
  summarise(se = std.error(RMSE), RMSE = mean(RMSE)) %>% as.data.frame

library(ggplot2)
cmap = c(model_4_XGB = "#417ab2", 
         model_3_RF = "#ef8939",
         model_5_LGBM = "#57a245",
         model_2_SVR = "#c63f38",
         model_1_Ridge = "#906cb9",
         model_1_ElasticNet = "#8c645a",
         model_1_Lasso = "#d680c0")
cbreaks = c("model_4_XGB", "model_3_RF", "model_5_LGBM", "model_2_SVR", "model_1_Ridge", "model_1_ElasticNet", "model_1_Lasso")
clabels = c("XGBoost", "RandomForest", "LigtGBM", "SVR", "Ridge Regression", "Elastic Net", "LASSO Regression")
library(extrafont)
p_line <- ggplot(data = df_models_g, aes(x = x, y = RMSE, fill = Model)) +
  geom_line(aes(colour = Model)) +
  geom_point(aes(color = Model)) +
  geom_ribbon(aes(ymin = RMSE - se, ymax = RMSE + se), alpha = 0.3) +
  scale_color_manual(values = cmap, breaks = cbreaks, labels = clabels) +
  scale_fill_manual(values = cmap, breaks = cbreaks, labels = clabels) +
  labs(x = "Training Sample Size") +
  theme_bw() + 
  theme(panel.grid = element_blank(),
        legend.position = 'none',
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 12),
        # plot.margin = margin(r=20)
        )

ggsave(p_line, file = "./figures/final/2_robust_line_RMSE_no_legend.pdf", family = "ArialMT", width = 4.5, height = 5,
      )

  df_se <- df_models_g %>% group_by(Model) %>% summarise(se=mean(se)) %>% as.data.frame
write.csv(df_se, "./figures/final/2_robust_line_RMSE.csv", row.names = F)
