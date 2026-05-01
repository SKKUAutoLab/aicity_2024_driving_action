from setuptools import setup, find_packages

setup(name="OpenAnomaly", version="1.0", author="TRAN DAI CHI", author_email="ctran743@gmail.com", description="README.md", url="", packages=find_packages(exclude=["envs*"]),
      py_modules=["AICity_AD_Models", "Dashcam_AD_Models", "Extension_Research_Models", "Feature_Extraction_Models", "Industrial_AD_Models", "Surveillance_AD_Models", "Unsupervised_AD_Models", "WSAD_Models"],
      license="LICENSE", python_requires=">=3.8", include_package_data=True, install_requires="requirements.txt")