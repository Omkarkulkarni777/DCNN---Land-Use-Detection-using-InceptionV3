Land Use Classification with InceptionV3 and GLCM Features

This project aims to classify land use and land cover (LULC) types using deep learning and traditional machine learning techniques. The approach involves extracting features from images using the InceptionV3 model and combining them with GLCM (Gray Level Co-occurrence Matrix) features to improve classification performance.
Key Steps:

    InceptionV3 Feature Extraction:
    Pre-trained on ImageNet, the InceptionV3 model is fine-tuned to extract features from input images. These features capture important visual patterns that are critical for land use classification.

    GLCM Feature Extraction:
    GLCM features, such as contrast, correlation, and homogeneity, are computed from the images to capture texture-based patterns.

    Feature Aggregation:
    The InceptionV3 features and GLCM features are extracted separately and appended into a single dataset. This combined feature set provides richer information for classification.

    Modeling with Weka:
    The aggregated features are exported into an Excel file for further analysis. These features are then imported into Weka, a machine learning tool, for classification. Several machine learning algorithms are applied, including:
        LMT (Logistic Model Tree)
        Random Forest (RF)
        Regression Trees (RT)
        Decision Tree Classifier

    Results:
    The performance of each classifier is evaluated and compared to determine the most effective model for land use classification.

Tools & Technologies:

    TensorFlow for feature extraction using InceptionV3.
    GLCM for texture analysis.
    Weka for machine learning and classification.
