import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_path = ("/Users/cocoweng/Desktop/database/Step 1 Processing Data/Tourism/Data/STI data.xlsx")
df_traditional = pd.read_excel(file_path)

pca_traditional_results = []
n_components_list = {}  

years = df_traditional["Year"].unique()

for year in years:
    df_year = df_traditional[df_traditional["Year"] == year].copy()
    provinces = df_year["Province"].values
    df_year_features = df_year.drop(columns=["Province", "Year"])

    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_year_features)
    
    pca_full = PCA()
    pca_full.fit(scaled_data)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    n_components = np.argmax(cumulative_variance >= 0.85) + 1
    n_components_list[year] = n_components 

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)

    explained_variance = pca.explained_variance_ratio_
    index_scores = np.sum(principal_components * explained_variance, axis=1)

    temp_df = pd.DataFrame({
        "Province": provinces,
        "Year": year,
        "STIndex": index_scores
    })

    pca_traditional_results.append(temp_df)


pca_traditional_index_df = pd.concat(pca_traditional_results, ignore_index=True)
n_components_df = pd.DataFrame(list(n_components_list.items()), columns=["Year", "Number of principal components"])

output_path_index = "/Users/cocoweng/Desktop/database/Step 1 Processing Data/Tourism/STI PCA Result/PCA_STI_Index.xlsx"
output_path_components = "/Users/cocoweng/Desktop/database/Step 1 Processing Data/Tourism/STI PCA Result/PCA_STI_Components.xlsx"

pca_traditional_index_df.to_excel(output_path_index, index=False)
n_components_df.to_excel(output_path_components, index=False)


