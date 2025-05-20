# 00-machine-learning-full-stack


Si deseas crear las mismas carpetas para crear tu repositorio y prácticar utiliza este código. Solo copialo y pégalo
en el terminal

```
# ----------- BEGIN COPY --------------
root="01-machine-learning-full-stack"

read -r -d '' MODELS <<'EOF'
01-Supervised_Learning/Regression/01-Linear_Regression
01-Supervised_Learning/Regression/02-Polynomial_Regression
01-Supervised_Learning/Regression/03-Ridge_Regression
01-Supervised_Learning/Regression/04-Lasso_Regression
01-Supervised_Learning/Regression/05-Elastic_Net
01-Supervised_Learning/Regression/10-SVR
01-Supervised_Learning/Classification/06-Logistic_Regression
01-Supervised_Learning/Classification/07-SGD_Classifier
01-Supervised_Learning/Classification/08-Passive_Aggressive
01-Supervised_Learning/Classification/09-Support_Vector_Machines
01-Supervised_Learning/Classification/11-K_Nearest_Neighbors
01-Supervised_Learning/Classification/12-Decision_Trees
01-Supervised_Learning/Classification/13-Random_Forest
01-Supervised_Learning/Classification/14-Extra_Trees
01-Supervised_Learning/Classification/15-Extremely_Randomized_Trees
01-Supervised_Learning/Classification/16-Gradient_Boosting
01-Supervised_Learning/Classification/17-AdaBoost
01-Supervised_Learning/Classification/18-XGBoost
01-Supervised_Learning/Classification/19-LightGBM
01-Supervised_Learning/Classification/20-CatBoost
01-Supervised_Learning/Classification/21-Bagging_Classifier
01-Supervised_Learning/Classification/22-Voting_Classifier
01-Supervised_Learning/Classification/23-Stacking_Classifier
01-Supervised_Learning/Classification/24-Naive_Bayes
02-Unsupervised_Learning/Clustering/01-K_Means
02-Unsupervised_Learning/Clustering/02-Spectral_Clustering
02-Unsupervised_Learning/Clustering/03-Affinity_Propagation
02-Unsupervised_Learning/Clustering/04-Hierarchical_Clustering
02-Unsupervised_Learning/Clustering/05-DBSCAN
02-Unsupervised_Learning/Clustering/06-OPTICS
02-Unsupervised_Learning/Clustering/07-Gaussian_Mixture_Model
02-Unsupervised_Learning/Anomaly_Detection/08-Isolation_Forest
02-Unsupervised_Learning/Anomaly_Detection/09-LOF
02-Unsupervised_Learning/Dimensionality_Reduction/10-PCA
02-Unsupervised_Learning/Dimensionality_Reduction/11-NMF
02-Unsupervised_Learning/Dimensionality_Reduction/12-t_SNE
02-Unsupervised_Learning/Dimensionality_Reduction/13-UMAP
02-Unsupervised_Learning/Dimensionality_Reduction/14-Self_Organizing_Map
02-Unsupervised_Learning/Dimensionality_Reduction/15-Autoencoders
EOF

IFS=$'\n'
for line in $MODELS; do
  base="$root/$line"
  name="${base##*/}"
  name="${name:3}"                       # quita prefijo numérico
  for d in 01-Theory 02-Implementation 03-Evaluation 04-Deployment 05-MLOps; do
    mkdir -p "$base/$d"
  done

  touch "$base/01-Theory/$name.md" \
        "$base/02-Implementation/$name.ipynb" \
        "$base/03-Evaluation/Model_Evaluation.ipynb" \
        "$base/03-Evaluation/Hyperparameter_Tuning.ipynb" \
        "$base/04-Deployment/Model_Deployment.md" \
        "$base/04-Deployment/Serving_with_FastAPI.ipynb" \
        "$base/04-Deployment/Containerization_with_Docker.md" \
        "$base/05-MLOps/MLOps_Pipeline.md" \
        "$base/05-MLOps/Model_Monitoring.md" \
        "$base/05-MLOps/Automatic_Retraining_Pipeline.ipynb"
done
unset IFS
echo "✅  Estructura creada en $(pwd)/$root"
# ----------- END COPY --------------

```