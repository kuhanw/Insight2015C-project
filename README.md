# FeatureRank

This was the consulting project I did for an ad-tech startup in NYC as a part of my time in the Insight Data Science fellowship in Fall of 2015. The goal was to predict how to optimally place advertisements on select websites based on their content. 

I built an end-to-end pipeline that extracted NLP based features from the text of the webpage that the advertisement was embedded on and fed them into a number of different classifiers to perform prediction. The target being the degree of user engagement with the target ad. 

Looking back, many of the features I looked at were quite simple, 
  - TF-IDF token counts,
  - n-grams,
  - token lengths,
  
despite that the model returned good results. The key was turning the problem from regression to classification. The original intent was to predict the number of clicks that a advertisement would receive given the content of the page it was placed on. That was too ambitious for the amount, quality of data I had access to. It was however quite possible to predict with good accuracy if an advertisement would be clicked or not.

As this was a real consulting project for a startup, I have removed all the private datasets I used to perform the analysis
but the full write up of the pipeline is here: http://kuhanw.zohosites.com/. The pipeline code shown here is what I shipped to the company 
to be used in production.
