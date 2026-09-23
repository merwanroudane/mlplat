"""Reference registry.

Every DOI below was resolved on doi.org AND its title/author matched on
Crossref on 2026-09-23; every arXiv id was matched to its title on arxiv.org
the same day (see docs/research_notes.md). Nothing here is from memory alone.
Pages cite by key: ``cite("chernozhukov2018")``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    key: str
    citation: str
    link: str         # "doi:..." | "arXiv:..." | "https://..."
    topics: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        if self.link.startswith("doi:"):
            return "https://doi.org/" + self.link[4:]
        if self.link.startswith("arXiv:"):
            return "https://arxiv.org/abs/" + self.link[6:]
        return self.link


_R = Ref

REFERENCES: tuple[Ref, ...] = (
    # ---------------------------------------------------------------- books
    _R("esl", "Hastie, T., Tibshirani, R. & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.",
       "doi:10.1007/978-0-387-84858-7", ("foundations", "supervised", "trees_ensembles")),
    _R("islp", "James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). An Introduction to Statistical "
               "Learning with Applications in Python. Springer.", "doi:10.1007/978-3-031-38747-0", ("foundations",)),
    _R("uml", "Shalev-Shwartz, S. & Ben-David, S. (2014). Understanding Machine Learning: From Theory to Algorithms. "
              "Cambridge University Press.", "doi:10.1017/CBO9781107298019", ("learning_theory",)),
    _R("imbens_rubin", "Imbens, G. W. & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical "
                       "Sciences. Cambridge University Press.", "doi:10.1017/CBO9781139025751", ("causal_ml",)),
    _R("vovk2005", "Vovk, V., Gammerman, A. & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.",
       "doi:10.1007/b106715", ("conformal_prediction",)),
    _R("mitchell1997", "Mitchell, T. M. (1997). Machine Learning. McGraw-Hill. ISBN 0-07-042807-7.",
       "https://openlibrary.org/isbn/0070428077", ("what_is_ml",)),
    _R("cs229","Stanford CS229: Machine Learning — course notes and syllabus.", "https://cs229.stanford.edu/",
       ("curriculum",)),
    _R("mlcc", "Google Machine Learning Crash Course.", "https://developers.google.com/machine-learning/crash-course",
       ("curriculum", "production")),
    # ------------------------------------------------------ linear models
    _R("hoerl1970", "Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal "
                    "problems. Technometrics, 12(1), 55–67.", "doi:10.1080/00401706.1970.10488634", ("regularization",)),
    _R("tibshirani1996", "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. JRSS-B, 58(1), 267–288.",
       "doi:10.1111/j.2517-6161.1996.tb02080.x", ("regularization",)),
    _R("zou2005", "Zou, H. & Hastie, T. (2005). Regularization and variable selection via the elastic net. JRSS-B, 67(2), "
                  "301–320.", "doi:10.1111/j.1467-9868.2005.00503.x", ("regularization",)),
    _R("friedman2010", "Friedman, J., Hastie, T. & Tibshirani, R. (2010). Regularization paths for generalized linear "
                       "models via coordinate descent. Journal of Statistical Software, 33(1).",
       "doi:10.18637/jss.v033.i01", ("advanced_optimizers", "regularization")),
    _R("huber1964", "Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics, "
                    "35(1), 73–101.", "doi:10.1214/aoms/1177703732", ("robust_regression", "loss_functions")),
    _R("fisher1936", "Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, "
                     "7(2), 179–188.", "doi:10.1111/j.1469-1809.1936.tb02137.x", ("lda_qda",)),
    _R("cover1967", "Cover, T. & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Trans. Information "
                    "Theory, 13(1), 21–27.", "doi:10.1109/TIT.1967.1053964", ("knn",)),
    _R("cortes1995", "Cortes, C. & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20, 273–297.",
       "doi:10.1007/BF00994018", ("svm",)),
    # -------------------------------------------------------- trees/ensembles
    _R("quinlan1986", "Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1, 81–106.",
       "doi:10.1007/BF00116251", ("decision_trees",)),
    _R("breiman1996", "Breiman, L. (1996). Bagging predictors. Machine Learning, 24, 123–140.", "doi:10.1007/BF00058655",
       ("ensemble_learning",)),
    _R("breiman2001", "Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.", "doi:10.1023/A:1010933404324",
       ("random_forest", "model_inspection")),
    _R("geurts2006", "Geurts, P., Ernst, D. & Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63, 3–42.",
       "doi:10.1007/s10994-006-6226-1", ("random_forest",)),
    _R("freund1997", "Freund, Y. & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and "
                     "an application to boosting. JCSS, 55(1), 119–139.", "doi:10.1006/jcss.1997.1504", ("boosting",)),
    _R("friedman2001", "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of "
                       "Statistics, 29(5), 1189–1232.", "doi:10.1214/aos/1013203451", ("boosting", "pdp_ice")),
    _R("chen2016", "Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16, 785–794.",
       "doi:10.1145/2939672.2939785", ("xgboost",)),
    _R("ke2017", "Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 30.",
       "https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree", ("lightgbm",)),
    _R("prokhorenkova2018", "Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. "
                            "NeurIPS 31.", "arXiv:1706.09516", ("catboost",)),
    # ------------------------------------------------------------ optimisation
    _R("robbins1951", "Robbins, H. & Monro, S. (1951). A stochastic approximation method. Annals of Mathematical "
                      "Statistics, 22(3), 400–407.", "doi:10.1214/aoms/1177729586", ("gradient_descent",)),
    _R("liu1989", "Liu, D. C. & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. "
                  "Mathematical Programming, 45, 503–528.", "doi:10.1007/BF01589116", ("advanced_optimizers",)),
    _R("kingma2015", "Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.",
       "arXiv:1412.6980", ("advanced_optimizers", "dl_bridge")),
    _R("rumelhart1986", "Rumelhart, D. E., Hinton, G. E. & Williams, R. J. (1986). Learning representations by "
                        "back-propagating errors. Nature, 323, 533–536.", "doi:10.1038/323533a0", ("dl_bridge",)),
    # --------------------------------------------------- validation & theory
    _R("efron1979", "Efron, B. (1979). Bootstrap methods: Another look at the jackknife. Annals of Statistics, 7(1), 1–26.",
       "doi:10.1214/aos/1176344552", ("statistics_ml",)),
    _R("varma2006", "Varma, S. & Simon, R. (2006). Bias in error estimation when using cross-validation for model "
                    "selection. BMC Bioinformatics, 7, 91.", "doi:10.1186/1471-2105-7-91", ("cross_validation",)),
    _R("nadeau2003", "Nadeau, C. & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52, "
                     "239–281.", "doi:10.1023/A:1024068626366", ("cross_validation", "model_comparison")),
    _R("kaufman2012", "Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining: Formulation, "
                      "detection, and avoidance. ACM TKDD, 6(4).", "doi:10.1145/2382577.2382579", ("leakage",)),
    _R("wolpert1996", "Wolpert, D. H. (1996). The lack of a priori distinctions between learning algorithms. Neural "
                      "Computation, 8(7), 1341–1390.", "doi:10.1162/neco.1996.8.7.1341", ("learning_theory",)),
    _R("shmueli2010", "Shmueli, G. (2010). To explain or to predict? Statistical Science, 25(3), 289–310.",
       "doi:10.1214/10-STS330", ("prediction_vs_causality",)),
    _R("mullainathan2017", "Mullainathan, S. & Spiess, J. (2017). Machine learning: An applied econometric approach. "
                           "Journal of Economic Perspectives, 31(2), 87–106.", "doi:10.1257/jep.31.2.87",
       ("prediction_vs_causality",)),
    _R("varian2014", "Varian, H. R. (2014). Big data: New tricks for econometrics. JEP, 28(2), 3–28.",
       "doi:10.1257/jep.28.2.3", ("prediction_vs_causality",)),
    _R("lei2018", "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. (2018). Distribution-free "
                  "predictive inference for regression. JASA, 113(523), 1094–1111.", "doi:10.1080/01621459.2017.1307116",
       ("conformal_prediction",)),
    _R("angelopoulos2021", "Angelopoulos, A. N. & Bates, S. (2021). A gentle introduction to conformal prediction and "
                           "distribution-free uncertainty quantification.", "arXiv:2107.07511", ("conformal_prediction",)),
    # ------------------------------------------------ evaluation/calibration
    _R("niculescu2005", "Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities with supervised "
                        "learning. ICML '05.", "doi:10.1145/1102351.1102430", ("calibration",)),
    _R("zadrozny2002", "Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate multiclass "
                       "probability estimates. KDD '02.", "doi:10.1145/775047.775151", ("calibration",)),
    _R("guo2017", "Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On calibration of modern neural networks. "
                  "ICML.", "arXiv:1706.04599", ("calibration",)),
    _R("elkan2001", "Elkan, C. (2001). The foundations of cost-sensitive learning. IJCAI 2001, 973–978.",
       "https://dblp.org/rec/conf/ijcai/Elkan01.html", ("threshold_tuning",)),
    _R("chawla2002", "Chawla, N. V. et al. (2002). SMOTE: Synthetic minority over-sampling technique. JAIR, 16, 321–357.",
       "doi:10.1613/jair.953", ("imbalanced",)),
    # -------------------------------------------------------------- HPO
    _R("bergstra2012", "Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter optimization. JMLR, 13, "
                       "281–305.", "https://www.jmlr.org/papers/v13/bergstra12a.html", ("hpo_foundations",)),
    _R("akiba2019", "Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). Optuna: A next-generation "
                    "hyperparameter optimization framework. KDD '19.", "doi:10.1145/3292500.3330701",
       ("bayesian_optimization",)),
    _R("li2018", "Li, L. et al. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. JMLR, "
                 "18(185), 1–52.", "https://jmlr.org/papers/v18/16-558.html", ("hpo_foundations",)),
    # -------------------------------------------------------- unsupervised
    _R("lloyd1982", "Lloyd, S. (1982). Least squares quantization in PCM. IEEE Trans. Information Theory, 28(2), 129–137.",
       "doi:10.1109/TIT.1982.1056489", ("kmeans",)),
    _R("ward1963", "Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. JASA, 58(301), 236–244.",
       "doi:10.1080/01621459.1963.10500845", ("hierarchical",)),
    _R("ester1996", "Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. (1996). A density-based algorithm for discovering "
                    "clusters in large spatial databases with noise. KDD-96.", "https://cdn.aaai.org/KDD/1996/KDD96-037.pdf",
       ("dbscan",)),
    _R("campello2013", "Campello, R. J. G. B., Moulavi, D. & Sander, J. (2013). Density-based clustering based on "
                       "hierarchical density estimates. PAKDD 2013.", "doi:10.1007/978-3-642-37456-2_14", ("dbscan",)),
    _R("dempster1977", "Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum likelihood from incomplete data via "
                       "the EM algorithm. JRSS-B, 39(1), 1–38.", "doi:10.1111/j.2517-6161.1977.tb01600.x", ("gmm",)),
    _R("pearson1901", "Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. "
                      "Philosophical Magazine, 2(11), 559–572.", "doi:10.1080/14786440109462720", ("pca",)),
    _R("vandermaaten2008", "van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. JMLR, 9, 2579–2605.",
       "https://www.jmlr.org/papers/v9/vandermaaten08a.html", ("manifold",)),
    _R("mcinnes2018", "McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform manifold approximation and projection "
                      "for dimension reduction.", "arXiv:1802.03426", ("manifold",)),
    _R("liu2008", "Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). Isolation forest. ICDM 2008.",
       "doi:10.1109/ICDM.2008.17", ("anomaly_detection",)),
    _R("breunig2000", "Breunig, M. M., Kriegel, H.-P., Ng, R. T. & Sander, J. (2000). LOF: Identifying density-based "
                      "local outliers. SIGMOD 2000.", "doi:10.1145/335191.335388", ("anomaly_detection",)),
    _R("scholkopf2001", "Schölkopf, B. et al. (2001). Estimating the support of a high-dimensional distribution. Neural "
                        "Computation, 13(7), 1443–1471.", "doi:10.1162/089976601750264965", ("anomaly_detection",)),
    # -------------------------------------------------------- interpretability
    _R("goldstein2015", "Goldstein, A., Kapelner, A., Bleich, J. & Pitkin, E. (2015). Peeking inside the black box: "
                        "Visualizing statistical learning with plots of individual conditional expectation. JCGS, 24(1).",
       "doi:10.1080/10618600.2014.907095", ("pdp_ice",)),
    _R("lundberg2017", "Lundberg, S. M. & Lee, S.-I. (2017). A unified approach to interpreting model predictions. "
                       "NeurIPS 30.", "arXiv:1705.07874", ("shap",)),
    _R("ribeiro2016", "Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). \"Why should I trust you?\": Explaining the "
                      "predictions of any classifier. KDD '16.", "doi:10.1145/2939672.2939778", ("shap",)),
    # ------------------------------------------------------------- causal
    _R("rosenbaum1983", "Rosenbaum, P. R. & Rubin, D. B. (1983). The central role of the propensity score in "
                        "observational studies for causal effects. Biometrika, 70(1), 41–55.", "doi:10.1093/biomet/70.1.41",
       ("causal_foundations", "irm")),
    _R("robinson1988", "Robinson, P. M. (1988). Root-N-consistent semiparametric regression. Econometrica, 56(4), 931–954.",
       "doi:10.2307/1912705", ("plr", "dml_core")),
    _R("imbens1994", "Imbens, G. W. & Angrist, J. D. (1994). Identification and estimation of local average treatment "
                     "effects. Econometrica, 62(2), 467–475.", "doi:10.2307/2951620", ("iivm",)),
    _R("angrist1996", "Angrist, J. D., Imbens, G. W. & Rubin, D. B. (1996). Identification of causal effects using "
                      "instrumental variables. JASA, 91(434), 444–455.", "doi:10.1080/01621459.1996.10476902",
       ("iivm", "pliv")),
    _R("belloni2014", "Belloni, A., Chernozhukov, V. & Hansen, C. (2014). Inference on treatment effects after selection "
                      "among high-dimensional controls. Review of Economic Studies, 81(2), 608–650.",
       "doi:10.1093/restud/rdt044", ("dml_core",)),
    _R("belloni2017", "Belloni, A., Chernozhukov, V., Fernández-Val, I. & Hansen, C. (2017). Program evaluation and "
                      "causal inference with high-dimensional data. Econometrica, 85(1), 233–298.",
       "doi:10.3982/ECTA12723", ("irm", "dml_extensions")),
    _R("chernozhukov2018", "Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. & Robins, "
                           "J. (2018). Double/debiased machine learning for treatment and structural parameters. The "
                           "Econometrics Journal, 21(1), C1–C68.", "doi:10.1111/ectj.12097",
       ("dml_core", "neyman_orthogonality", "cross_fitting", "plr", "pliv", "irm", "iivm")),
    _R("bach2022", "Bach, P., Chernozhukov, V., Kurz, M. S. & Spindler, M. (2022). DoubleML — An object-oriented "
                   "implementation of double machine learning in Python. JMLR, 23(53), 1–6.", "arXiv:2104.03220",
       ("doubleml_package",)),
    _R("chiang2022", "Chiang, H. D., Kato, K., Ma, Y. & Sasaki, Y. (2022). Multiway cluster robust double/debiased "
                     "machine learning. JBES, 40(3), 1046–1056.", "doi:10.1080/07350015.2021.1895815", ("dml_extensions",)),
    _R("chernozhukov2022long", "Chernozhukov, V., Cinelli, C., Newey, W., Sharma, A. & Syrgkanis, V. (2022). Long story "
                               "short: Omitted variable bias in causal machine learning.", "arXiv:2112.13398",
       ("dml_extensions",)),
    _R("semenova2021", "Semenova, V. & Chernozhukov, V. (2021). Debiased machine learning of conditional average "
                       "treatment effects and other causal functions. The Econometrics Journal, 24(2), 264–289.",
       "doi:10.1093/ectj/utaa027", ("hte",)),
    _R("chang2020", "Chang, N.-C. (2020). Double/debiased machine learning for difference-in-differences models. The "
                    "Econometrics Journal, 23(2), 177–191.", "doi:10.1093/ectj/utaa001", ("panel_did",)),
    _R("santanna2020", "Sant'Anna, P. H. C. & Zhao, J. (2020). Doubly robust difference-in-differences estimators. "
                       "Journal of Econometrics, 219(1), 101–122.", "doi:10.1016/j.jeconom.2020.06.003", ("panel_did",)),
    _R("callaway2021", "Callaway, B. & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. "
                       "Journal of Econometrics, 225(2), 200–230.", "doi:10.1016/j.jeconom.2020.12.001", ("panel_did",)),
    _R("wager2018", "Wager, S. & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using "
                    "random forests. JASA, 113(523), 1228–1242.", "doi:10.1080/01621459.2017.1319839", ("hte",)),
    _R("athey2019", "Athey, S., Tibshirani, J. & Wager, S. (2019). Generalized random forests. Annals of Statistics, "
                    "47(2), 1148–1178.", "doi:10.1214/18-AOS1709", ("hte",)),
    _R("athey2016", "Athey, S. & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. PNAS, "
                    "113(27), 7353–7360.", "doi:10.1073/pnas.1510489113", ("hte",)),
    _R("kunzel2019", "Künzel, S. R., Sekhon, J. S., Bickel, P. J. & Yu, B. (2019). Metalearners for estimating "
                     "heterogeneous treatment effects using machine learning. PNAS, 116(10), 4156–4165.",
       "doi:10.1073/pnas.1804597116", ("hte",)),
    _R("nie2021", "Nie, X. & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, "
                  "108(2), 299–319.", "doi:10.1093/biomet/asaa076", ("hte",)),
    _R("kennedy2023", "Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. "
                      "Electronic Journal of Statistics, 17(2), 3008–3049.", "doi:10.1214/23-EJS2157", ("hte",)),
    # ---------------------------------------------------- modern/production
    _R("watkins1992", "Watkins, C. J. C. H. & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279–292.",
       "doi:10.1007/BF00992698", ("rl_bridge",)),
    _R("sutton2018", "Sutton, R. S. & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
       "http://incompleteideas.net/book/the-book-2nd.html", ("rl_bridge",)),
    _R("gama2014", "Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). A survey on concept "
                   "drift adaptation. ACM Computing Surveys, 46(4).", "doi:10.1145/2523813", ("drift", "modern_topics")),
    _R("rabanser2019", "Rabanser, S., Günnemann, S. & Lipton, Z. C. (2019). Failing loudly: An empirical study of "
                       "methods for detecting dataset shift. NeurIPS 32.", "arXiv:1810.11953", ("drift",)),
    _R("sculley2015", "Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. NeurIPS 28.",
       "https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems", ("production_ml", "mlops")),
    _R("mitchell2019", "Mitchell, M. et al. (2019). Model cards for model reporting. FAT* '19.",
       "doi:10.1145/3287560.3287596", ("fairness", "production_ml")),
    _R("hardt2016", "Hardt, M., Price, E. & Srebro, N. (2016). Equality of opportunity in supervised learning. NeurIPS 29.",
       "arXiv:1610.02413", ("fairness",)),
    _R("chouldechova2017", "Chouldechova, A. (2017). Fair prediction with disparate impact. Big Data, 5(2), 153–163.",
       "doi:10.1089/big.2016.0047", ("fairness",)),
    _R("kleinberg2016", "Kleinberg, J., Mullainathan, S. & Raghavan, M. (2016). Inherent trade-offs in the fair "
                        "determination of risk scores.", "arXiv:1609.05807", ("fairness",)),
    _R("pedregosa2011", "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825–2830.",
       "https://jmlr.org/papers/v12/pedregosa11a.html", ("pipelines",)),
    # ------------------------------------------------ Leo Breiman: philosophy and contributions
    _R("breiman2001two", "Breiman, L. (2001). Statistical modeling: The two cultures (with comments by D. R. Cox, B. Efron, "
                         "B. Hoadley, E. Parzen and a rejoinder by the author). Statistical Science, 16(3), 199–231.",
       "doi:10.1214/ss/1009213726", ("breiman",)),
    _R("breiman1984", "Breiman, L., Friedman, J. H., Olshen, R. A. & Stone, C. J. (1984). Classification and Regression "
                      "Trees. Wadsworth (reprinted by Chapman & Hall/CRC, 2017).", "doi:10.1201/9781315139470",
       ("breiman", "trees_ensembles")),
    _R("breiman1985ace", "Breiman, L. & Friedman, J. H. (1985). Estimating optimal transformations for multiple regression "
                         "and correlation. JASA, 80(391), 580–598.", "doi:10.1080/01621459.1985.10478157", ("breiman",)),
    _R("breiman1995garrote", "Breiman, L. (1995). Better subset regression using the nonnegative garrote. Technometrics, "
                             "37(4), 373–384.", "doi:10.1080/00401706.1995.10484371", ("breiman", "regularization")),
    _R("breiman1996stack", "Breiman, L. (1996). Stacked regressions. Machine Learning, 24, 49–64.",
       "doi:10.1007/BF00117832", ("breiman", "trees_ensembles")),
    _R("breiman1998arcing", "Breiman, L. (1998). Arcing classifiers (with discussion and a rejoinder by the author). "
                            "Annals of Statistics, 26(3), 801–849.", "doi:10.1214/aos/1024691079", ("breiman", "boosting")),
    _R("efron2020", "Efron, B. (2020). Prediction, estimation, and attribution. JASA, 115(530), 636–655.",
       "doi:10.1080/01621459.2020.1762613", ("breiman",)),
    _R("rudin2019", "Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use "
                    "interpretable models instead. Nature Machine Intelligence, 1, 206–215.",
       "doi:10.1038/s42256-019-0048-x", ("breiman", "interpretability")),
    _R("semenova2022", "Semenova, L., Rudin, C. & Parr, R. (2022). On the existence of simpler machine learning models. "
                       "FAccT '22, 1827–1858.", "doi:10.1145/3531146.3533232", ("breiman",)),
    _R("fisher2019", "Fisher, A., Rudin, C. & Dominici, F. (2019). All models are wrong, but many are useful: Learning a "
                     "variable's importance by studying an entire class of prediction models simultaneously. JMLR, 20(177).",
       "https://www.jmlr.org/papers/v20/18-760.html", ("breiman", "interpretability")),
    # ------------------------------------------------ imbalanced classification
    _R("he2009", "He, H. & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data "
                 "Engineering, 21(9), 1263–1284.", "doi:10.1109/TKDE.2008.239", ("imbalanced",)),
    _R("han2005", "Han, H., Wang, W.-Y. & Mao, B.-H. (2005). Borderline-SMOTE: A new over-sampling method in imbalanced "
                  "data sets learning. ICIC 2005, LNCS 3644, 878–887.", "doi:10.1007/11538059_91", ("imbalanced",)),
    _R("he2008adasyn", "He, H., Bai, Y., Garcia, E. A. & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for "
                       "imbalanced learning. IJCNN 2008, 1322–1328.", "doi:10.1109/IJCNN.2008.4633969", ("imbalanced",)),
    _R("tomek1976", "Tomek, I. (1976). Two modifications of CNN. IEEE Transactions on Systems, Man, and Cybernetics, "
                    "SMC-6(11), 769–772.", "doi:10.1109/TSMC.1976.4309452", ("imbalanced",)),
    _R("wilson1972", "Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules using edited data. IEEE "
                     "Transactions on Systems, Man, and Cybernetics, SMC-2(3), 408–421.", "doi:10.1109/TSMC.1972.4309137",
       ("imbalanced",)),
    _R("hart1968", "Hart, P. (1968). The condensed nearest neighbor rule. IEEE Transactions on Information Theory, 14(3), "
                   "515–516.", "doi:10.1109/TIT.1968.1054155", ("imbalanced",)),
    _R("batista2004", "Batista, G. E. A. P. A., Prati, R. C. & Monard, M. C. (2004). A study of the behavior of several "
                      "methods for balancing machine learning training data. ACM SIGKDD Explorations, 6(1), 20–29.",
       "doi:10.1145/1007730.1007735", ("imbalanced",)),
    _R("liu2009", "Liu, X.-Y., Wu, J. & Zhou, Z.-H. (2009). Exploratory undersampling for class-imbalance learning. IEEE "
                  "Transactions on Systems, Man, and Cybernetics, Part B, 39(2), 539–550.", "doi:10.1109/TSMCB.2008.2007853",
       ("imbalanced",)),
    _R("seiffert2010", "Seiffert, C., Khoshgoftaar, T. M., Van Hulse, J. & Napolitano, A. (2010). RUSBoost: A hybrid "
                       "approach to alleviating class imbalance. IEEE Transactions on Systems, Man, and Cybernetics, Part A, "
                       "40(1), 185–197.", "doi:10.1109/TSMCA.2009.2029559", ("imbalanced",)),
    _R("chen2004brf", "Chen, C., Liaw, A. & Breiman, L. (2004). Using random forest to learn imbalanced data. Technical "
                      "Report 666, Department of Statistics, UC Berkeley.",
       "https://statistics.berkeley.edu/tech-reports/666", ("imbalanced", "breiman")),
    _R("saerens2002", "Saerens, M., Latinne, P. & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new "
                      "a priori probabilities: A simple procedure. Neural Computation, 14(1), 21–41.",
       "doi:10.1162/089976602753284446", ("imbalanced", "calibration")),
    _R("vandengoorbergh2022", "van den Goorbergh, R., van Smeden, M., Timmerman, D. & Van Calster, B. (2022). The harm of "
                              "class imbalance corrections for risk prediction models. JAMIA, 29(9), 1525–1534.",
       "doi:10.1093/jamia/ocac093", ("imbalanced", "calibration")),
    _R("lin2017focal", "Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P. (2017). Focal loss for dense object "
                       "detection. ICCV 2017.", "arXiv:1708.02002", ("imbalanced",)),
    _R("lemaitre2017", "Lemaître, G., Nogueira, F. & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the "
                       "curse of imbalanced datasets in machine learning. JMLR, 18(17), 1–5.",
       "https://www.jmlr.org/papers/v18/16-365.html", ("imbalanced",)),
)

REF_BY_KEY: dict[str, Ref] = {r.key: r for r in REFERENCES}


def cite(*keys: str) -> str:
    """Markdown list of references for the given keys (used at the bottom of pages)."""
    return "\n".join(f"- {REF_BY_KEY[k].citation} [{REF_BY_KEY[k].link}]({REF_BY_KEY[k].url})" for k in keys)


# Official documentation pages consulted, with the version verified (section 6)
OFFICIAL_DOCS: tuple[tuple[str, str, str], ...] = (
    ("scikit-learn", "1.9.1", "https://scikit-learn.org/stable/whats_new/v1.9.html"),
    ("scikit-learn 1.8 changelog", "1.8", "https://scikit-learn.org/stable/whats_new/v1.8.html"),
    ("Streamlit", "1.64.0", "https://docs.streamlit.io/develop/quick-reference/release-notes"),
    ("XGBoost parameters", "3.2.0 / 3.4.1", "https://xgboost.readthedocs.io/en/stable/parameter.html"),
    ("LightGBM parameters", "4.7.0", "https://lightgbm.readthedocs.io/en/stable/Parameters.html"),
    ("CatBoost training parameters", "1.2.10", "https://catboost.ai/docs/en/references/training-parameters/"),
    ("Optuna", "5.0.0", "https://optuna.readthedocs.io/en/stable/"),
    ("SHAP", "0.51.0 / 0.52.0", "https://shap.readthedocs.io/en/latest/"),
    ("imbalanced-learn", "0.14.2", "https://imbalanced-learn.org/stable/"),
    ("DoubleML", "0.11.4", "https://docs.doubleml.org/stable/"),
    ("EconML", "0.17.0", "https://www.pywhy.org/EconML/"),
    ("DoWhy", "0.14", "https://www.pywhy.org/dowhy/"),
    ("MLflow", "3.16.1", "https://mlflow.org/docs/latest/"),
    ("Plotly", "7.1.0", "https://plotly.com/python/"),
    ("pandas", "3.0.6", "https://pandas.pydata.org/docs/"),
    ("statsmodels", "0.15.0", "https://www.statsmodels.org/stable/"),
)
