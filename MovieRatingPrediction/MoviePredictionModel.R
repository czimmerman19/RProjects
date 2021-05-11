##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

options(digits=5) 

options(dplyr.summarise.inform=F) 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tibble)) install.packages("tibble", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(tibble)
library(lubridate)
library(matrixStats)
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#clear the environment.
  rm(list=ls())

  RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }

  start_time<-Sys.time()
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
gc()

print("data downloaded, begin model development code")
mu<-mean(edx$rating) #mean of all ratings

# to test/tune regularization parameters
lambda_seq<-seq(0, .02, .01)

#Note that a much higher range of lambda parameters was evaluated in exploratory 
#research. Using a range that only includes final ones chosend to speed up code.
k_seq<-seq(1,5)
k<-5

#create all of the folds up front.  Use the same k-fold partition
#of the movie set for all of the validation of the tuning parameters.
set.seed(989, sample.kind="Rounding")
cv_indexes <- createFolds(1:nrow(edx), k, list = FALSE)
#add all the additional variables required for the model:
edx<-edx %>%
  mutate(Drama=ifelse(str_detect(genres, 'Drama')==1, 1,0)) %>% 
  mutate(Comedy=ifelse(str_detect(genres, 'Comedy')==1, 1,0)) %>%  
  mutate(Thriller=ifelse(str_detect(genres, 'Thriller')==1, 1,0)) %>% 
  mutate(Romance=ifelse(str_detect(genres, 'Romance')==1, 1,0)) %>% 
  mutate(Children=ifelse(str_detect(genres, 'Children')==1, 1,0)) %>% 
  mutate(SciFi=ifelse(str_detect(genres, 'Sci-Fi')==1, 1,0)) %>%   
  mutate(Action=ifelse(str_detect(genres, 'Action')==1, 1,0)) %>%
  mutate(year_rated = year(as_datetime(timestamp))) 

cv_movies_1  <- edx[cv_indexes==1]
cv_movies_2 <- edx[cv_indexes==2]
cv_movies_3 <- edx[cv_indexes==3]
cv_movies_4 <- edx[cv_indexes==4]
cv_movies_5 <- edx[cv_indexes==5]
cv_movie_list<-list(cv_movies_1, cv_movies_2, cv_movies_3, cv_movies_4, cv_movies_5)


mu_cvs<-sapply(k_seq, function(k)
{
  mu_cv<-mean(cv_movie_list[[k]]$rating)
}
)
#start by predicting the mean for every rating to compare the results
#to later steps in the model.
base_rmse<-RMSE(edx$rating, mu)
print(paste("base rmse, using mean as predictor across the board", base_rmse))

  print("Tuning parameter for movie effect")
  cv_rmses<-sapply(lambda_seq, function(l) {
   
    rmses<-sapply(k_seq, function(fold)
    {
      print(paste("checking lambda", l, "fold", fold))
      cv_movies<- cv_movie_list[[fold]]
      mu_cv<-mu_cvs[fold]
      cv_movie_avgs<- cv_movies %>% 
        group_by(movieId) %>% 
        summarize(b_i = sum(rating - mu_cv)/(n()+l))
      
      cv_predict <- cv_movies %>%
        left_join(cv_movie_avgs, by='movieId') %>% 
        mutate(pred=mu_cv + b_i) %>% 
        mutate(pred=ifelse(pred<.05, .05, pred)) %>%
        mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>% pull(pred)                                         
      
      rmse<-RMSE(cv_movies$rating, cv_predict) 
      rmse
    }
    )
    mean(rmses)
    
  } 
  )   
  
  movie_effect_lambda<-lambda_seq[which.min(cv_rmses)]
  print(paste("movie effect lambda",  movie_effect_lambda))
  print(paste("lowest cross validated rmse for movie effect", cv_rmses[which.min(cv_rmses)]))
  
  #folds are constants so we can save off the avgs used to derive the weights
  #once we know the lambda.
cv_movie_avgs_list<- lapply(k_seq, function(k)
  {
    movie_avgs<- cv_movie_list[[k]] %>% 
      group_by(movieId) %>% 
      summarize(b_i = sum(rating - mu_cvs[k])/(n()+movie_effect_lambda))
    movie_avgs
  }
      
)
#staging now, to use in subsequent model evaluation steps:
edx_movie_avgs<- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+ movie_effect_lambda))

predicted_ratings <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  mutate(pred=mu+b_i)  %>%
  mutate(pred=ifelse(pred<.05, .05, pred)) %>%
  mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>%
  pull(pred)

rmse_movie_effect<-RMSE(edx$rating, predicted_ratings)
print(paste("movie effect rmse",  rmse_movie_effect))
print(paste("Model rmse improvement from user-genre component:", base_rmse-rmse_movie_effect))


  print("Tuning parameter for user_effect")
  
  cv_rmses<-sapply(lambda_seq, function(l) {
    
 
    rmses<-sapply(k_seq, function(fold)
    {
      print(paste("checking lambda", l, "fold", fold))
      mu_cv<-mu_cvs[fold]
      cv_movies<- cv_movie_list[[fold]]
      cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
    cv_user_avgs <- cv_movies %>% 
        left_join(cv_movie_avgs, by='movieId') %>%
        group_by(userId) %>%
        summarize(b_u = sum((rating - mu_cv - b_i))/(n()+l))

      cv_predict <- cv_movies %>%
        left_join(cv_movie_avgs, by='movieId') %>% 
        left_join(cv_user_avgs, by='userId') %>% 
        mutate(pred=mu_cv + b_i + b_u) %>% 
        mutate(pred=ifelse(pred<.05, .05, pred)) %>%
        mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>% pull(pred)                                         
      
      rmse<-RMSE(cv_movies$rating, cv_predict) 
      rmse
    }
    )
    mean(rmses)
    
  } 
  )
  user_effect_lambda<-lambda_seq[which.min(cv_rmses)]
  print(paste("user effect lambda",  user_effect_lambda))
  print(paste("lowest cross validated rmse for user effect", cv_rmses[which.min(cv_rmses)]))
  
  # save off the avgs used to derive the weights now that we have lambda
  cv_user_avgs_list<-lapply(k_seq, function(fold)
  {
    mu_cv<-mu_cvs[fold]
    cv_movies<- cv_movie_list[[fold]]
    cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
    cv_user_avgs <- cv_movies %>% 
      left_join(cv_movie_avgs, by='movieId') %>%
      group_by(userId) %>%
      summarize(b_u = sum((rating - mu_cv - b_i))/(n()+ user_effect_lambda))
  }
    )
  
   #save off the avgs used to derive the final predictions.
  edx_user_avgs <- edx %>% 
    left_join(edx_movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum((rating - mu - b_i))/(n()+user_effect_lambda))
  
  #calc rmse after user effect added:
  predicted_ratings <- edx %>%
    left_join(edx_movie_avgs, by='movieId') %>% 
    left_join(edx_user_avgs, by='userId') %>% 
      mutate(pred=mu+b_i + b_u)  %>%
    mutate(pred=ifelse(pred<.05, .05, pred)) %>%
    mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>%
    pull(pred)
  
  rmse_user_effect<-RMSE(edx$rating, predicted_ratings)
  print(paste("user effect rmse",  rmse_user_effect))
   print(paste("Model rmse improvement from user effect:", rmse_movie_effect-rmse_user_effect))
  
   print("Tuning parameter for genre effect")
  
  cv_rmses<-sapply(lambda_seq, function(l) {
    
    rmses<-sapply(k_seq, function(fold)
    {
      print(paste("checking lambda", l, "fold", fold))
      mu_cv<-mu_cvs[fold]
      cv_movies<- cv_movie_list[[fold]]
      cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
      cv_user_avgs <- cv_user_avgs_list[[fold]]
          cv_genre_avgs <- cv_movies %>% 
        left_join(cv_movie_avgs, by='movieId') %>%
        left_join(cv_user_avgs, by='userId') %>%
        group_by(genres) %>%
        summarize(b_g= sum((rating - mu_cv - b_i -b_u))/(n()+l))
 
           cv_predict <- cv_movies %>%
        left_join(cv_movie_avgs, by='movieId') %>% 
        left_join(cv_user_avgs, by='userId') %>% 
        left_join(cv_genre_avgs, by='genres') %>%
        mutate(pred=mu_cv + b_i + b_u + b_g) %>% 
        mutate(pred=ifelse(pred<.05, .05, pred)) %>%
        mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>% pull(pred)                                         
       
      rmse<-RMSE(cv_movies$rating, cv_predict) 
      rmse
    }
    )
    mean(rmses)
    
  } 
  )
 genre_effect_lambda<-lambda_seq[which.min(cv_rmses)]
 print(paste("genre effect lambda",  genre_effect_lambda))
 print(paste("lowest cross validated rmse for genre effect", cv_rmses[which.min(cv_rmses)]))
 
 #save off the avgs used for the x-validations below:
cv_genre_avgs_list<-lapply(k_seq, function(fold)
   {
     mu_cv<-mu_cvs[fold]
     cv_movies<- cv_movie_list[[fold]]
     cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
     cv_user_avgs <- cv_user_avgs_list[[fold]]
     cv_genre_avgs <- cv_movies %>% 
       left_join(cv_movie_avgs, by='movieId') %>%
       left_join(cv_user_avgs, by='userId') %>%
       group_by(genres) %>%
       summarize(b_g= sum((rating - mu_cv - b_i -b_u))/(n()+ genre_effect_lambda))

    } 
 )
 #save off the avgs used to derive the final predictions.
 edx_genre_avgs <- edx %>% 
   left_join(edx_movie_avgs, by='movieId') %>%
   left_join(edx_user_avgs, by='userId') %>%
   group_by(genres) %>%
   summarize(b_g = sum((rating - mu - b_i-b_u))/(n()+genre_effect_lambda))
 
 #check to see if genre_effect_adds to the model:
 predicted_ratings <- edx %>%
   left_join(edx_movie_avgs, by='movieId') %>% 
   left_join(edx_user_avgs, by='userId') %>% 
   left_join(edx_genre_avgs, by='genres') %>% 
   mutate(pred=mu+b_i + b_u + b_g)  %>%
   mutate(pred=ifelse(pred<.05, .05, pred)) %>%
   mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>%
   pull(pred)
 
 rmse_genre_effect<-RMSE(edx$rating, predicted_ratings)
 print(paste("Genre effect rmse",  rmse_genre_effect))
 print(paste("Model rmse improvement from genre component:", rmse_user_effect-rmse_genre_effect ))
 
print("Tuning parameter for user- genre effect")
cv_rmses<-sapply(lambda_seq, function(l) {
  print(paste("checking lambda", l))
  rmses<-sapply(k_seq, function(fold)
  {
    print(paste("fold", fold))
    mu_cv<-mu_cvs[fold]
    cv_movies<- cv_movie_list[[fold]]
    cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
    cv_user_avgs <- cv_user_avgs_list[[fold]]
    cv_genre_avgs <- cv_genre_avgs_list[[fold]]

    cv_user_drama_avgs <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      group_by(userId, Drama)%>%
      summarize(b_u_d = sum((rating - mu_cv - b_i-b_u - b_g))/(n()+l))
  
    cv_user_comedy_avgs <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      group_by(userId, Comedy)%>%
      summarize(b_u_c = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d))/(n()+l))
  
    cv_user_thriller_avgs <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      group_by(userId, Thriller)%>%
      summarize(b_u_t = sum((rating - mu_cv - b_i-b_u - b_g- b_u_d- b_u_c))/(n()+l))
  
    cv_user_romance_avgs <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
      group_by(userId, Romance)%>%
      summarize(b_u_r = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                               b_u_c- b_u_t))/(n()+l))
  
    cv_user_children_avgs <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
      left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
      group_by(userId, Children)%>%
      summarize(b_u_ch = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                                b_u_c - b_u_t- b_u_r))/(n()+l))
    
    cv_user_scifi_avgs <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
      left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
      left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
      group_by(userId, SciFi)%>%
      summarize(b_u_s = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                b_u_c - b_u_t- b_u_r - b_u_ch))/(n()+l))
    
    cv_user_action_avgs <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
      left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
      left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
      left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
      group_by(userId, Action)%>%
      summarize(b_u_a = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                               b_u_c - b_u_t- b_u_r - b_u_ch-b_u_s))/(n()+l))
    
  
    cv_predict <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
      left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
      left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
      left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
      left_join(cv_user_action_avgs, by=c('userId', 'Action'))  %>%
      mutate(pred=mu_cv + b_i + b_u + b_g+ b_u_d+ b_u_c+ 
               b_u_t+b_u_r+ b_u_s + b_u_ch + b_u_a) %>% 
      mutate(pred=ifelse(pred<.05, .05, pred)) %>%
      mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>% pull(pred) 
    
    rmse<-RMSE(cv_movies$rating, cv_predict) 
    rmse
  }
  )
  mean(rmses)
  
} 
)
user_genre_effect_lambda<-lambda_seq[which.min(cv_rmses)]
print(paste("user_genre_effect_lambda", user_genre_effect_lambda))
print(paste("lowest user_genre_effect", cv_rmses[which.min(cv_rmses)]))

#save off the avgs for each fold of user-genre-effect:
cv_user_drama_avgs_list<-lapply(k_seq, function(fold)
{
  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_movies %>%
    left_join(cv_movie_avgs, by='movieId') %>% 
    left_join(cv_user_avgs, by='userId') %>% 
    left_join(cv_genre_avgs, by='genres') %>%
    group_by(userId, Drama)%>%
    summarize(b_u_d = sum((rating - mu_cv - b_i-b_u - b_g))/(n()+user_genre_effect_lambda)) %>% 
    mutate_at(c(3), ~replace(., is.na(.), 0))
 
 } 
)
cv_user_comedy_avgs_list<-lapply(k_seq, function(fold)
{
  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
  cv_user_comedy_avgs <- cv_movies %>%
    left_join(cv_movie_avgs, by='movieId') %>% 
    left_join(cv_user_avgs, by='userId') %>% 
    left_join(cv_genre_avgs, by='genres') %>%
    left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
    group_by(userId, Comedy)%>%
    summarize(b_u_c = sum((rating - mu_cv - b_i
      -b_u - b_g - b_u_d))/(n()+user_genre_effect_lambda))  %>% 
    mutate_at(c(3), ~replace(., is.na(.), 0))
 

  
  } 
)

cv_user_thriller_avgs_list<-lapply(k_seq, function(fold)
{
  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
  cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]]
  cv_user_thriller_avgs <- cv_movies %>%
    left_join(cv_movie_avgs, by='movieId') %>% 
    left_join(cv_user_avgs, by='userId') %>% 
    left_join(cv_genre_avgs, by='genres') %>%
    left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
    left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
    group_by(userId, Thriller)%>%
    summarize(b_u_t = sum((rating - mu_cv - b_i-b_u -
        b_g- b_u_d- b_u_c))/(n()+user_genre_effect_lambda))%>% 
    mutate_at(c(3), ~replace(., is.na(.), 0))
   
} 
)
cv_user_romance_avgs_list<-lapply(k_seq, function(fold)
{
  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
  cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]]
  cv_user_thriller_avgs <- cv_user_thriller_avgs_list[[fold]]
  cv_user_romance_avgs <- cv_movies %>%
    left_join(cv_movie_avgs, by='movieId') %>% 
    left_join(cv_user_avgs, by='userId') %>% 
    left_join(cv_genre_avgs, by='genres') %>%
    left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
    left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
    left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
    group_by(userId, Romance)%>%
    summarize(b_u_r = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                             b_u_c- b_u_t))/(n()+user_genre_effect_lambda))%>% 
    mutate_at(c(3), ~replace(., is.na(.), 0))

} 
) 

cv_user_children_avgs_list<-lapply(k_seq, function(fold)
{

  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
  cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]]
  cv_user_thriller_avgs <- cv_user_thriller_avgs_list[[fold]]
  cv_user_romance_avgs <- cv_user_romance_avgs_list[[fold]]
  cv_user_children_avgs <- cv_movies %>%
    left_join(cv_movie_avgs, by='movieId') %>% 
    left_join(cv_user_avgs, by='userId') %>% 
    left_join(cv_genre_avgs, by='genres') %>%
    left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
    left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
    left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
    left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
    group_by(userId, Children) %>% 
    summarize(b_u_ch = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                              b_u_c - b_u_t- b_u_r))/(n()+user_genre_effect_lambda))%>% 
    mutate_at(c(3), ~replace(., is.na(.), 0))
} 
)




cv_user_scifi_avgs_list<-lapply(k_seq, function(fold)
{
  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
  cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]]
  cv_user_thriller_avgs <- cv_user_thriller_avgs_list[[fold]]
  cv_user_romance_avgs <- cv_user_romance_avgs_list[[fold]]
  cv_user_children_avgs <- cv_user_children_avgs_list[[fold]]
  cv_user_scifi_avgs <- cv_movies %>%
    left_join(cv_movie_avgs, by='movieId') %>% 
    left_join(cv_user_avgs, by='userId') %>% 
    left_join(cv_genre_avgs, by='genres') %>%
    left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
    left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
    left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
    left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
    left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
    group_by(userId, SciFi)%>%
    summarize(b_u_s = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                             b_u_c - b_u_t- b_u_r - b_u_ch))/(n()+user_genre_effect_lambda))%>% 
    mutate_at(c(3), ~replace(., is.na(.), 0))

} 
)
cv_user_action_avgs_list<-lapply(k_seq, function(fold)
{
  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
  cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]]
  cv_user_thriller_avgs <- cv_user_thriller_avgs_list[[fold]]
  cv_user_romance_avgs <- cv_user_romance_avgs_list[[fold]]
  cv_user_children_avgs <- cv_user_children_avgs_list[[fold]]
  cv_user_scifi_avgs <-cv_user_scifi_avgs_list[[fold]]
  cv_user_action_avgs <- cv_movies %>%
    left_join(cv_movie_avgs, by='movieId') %>% 
    left_join(cv_user_avgs, by='userId') %>% 
    left_join(cv_genre_avgs, by='genres') %>%
    left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
    left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
    left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
    left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
    left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
    left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
    group_by(userId, Action)%>%
    summarize(b_u_a = sum((rating - mu_cv - b_i-b_u - b_g - b_u_d- 
                             b_u_c - b_u_t- b_u_r - b_u_ch-b_u_s))/(n()+user_genre_effect_lambda))%>% 
    mutate_at(c(3), ~replace(., is.na(.), 0))
  
} 
)

gc()


#save off avgs to be used for prediction
edx_user_drama_avgs <-  edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>%
  group_by(userId, Drama)%>%
  summarize(b_u_d = sum((rating - mu - b_i-b_u - b_g))/(n()+user_genre_effect_lambda))

edx_user_comedy_avgs <-  edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>%
  left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
  group_by(userId, Comedy)%>%
  summarize(b_u_c = sum(rating - mu - b_i-b_u -
                          b_g-b_u_d)/(n()+user_genre_effect_lambda))

edx_user_thriller_avgs <-  edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>%
  left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
  left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
  group_by(userId, Thriller)%>%
  summarize(b_u_t = sum(rating - mu - b_i-b_u - b_g
                        -b_u_d-b_u_c)/(n()+user_genre_effect_lambda))

edx_user_romance_avgs <-  edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>%
  left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
  left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
  left_join(edx_user_thriller_avgs, by=c('userId','Thriller')) %>%
  group_by(userId, Romance)%>%
  summarize(b_u_r = sum(rating - mu - b_i-b_u - b_g-b_u_d
                        -b_u_c-b_u_t)/(n()+user_genre_effect_lambda))

edx_user_children_avgs <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>%
  left_join(edx_user_drama_avgs, by=c('userId', 'Drama'))  %>%
  left_join(edx_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
  left_join(edx_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
  left_join(edx_user_romance_avgs, by=c('userId', 'Romance'))  %>%
  group_by(userId, Children)%>%
  summarize(b_u_ch = sum((rating - mu - b_i-b_u - b_g - b_u_d- 
                            b_u_c - b_u_t- b_u_r))/(n()+user_genre_effect_lambda))

edx_user_scifi_avgs <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>%
  left_join(edx_user_drama_avgs, by=c('userId', 'Drama'))  %>%
  left_join(edx_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
  left_join(edx_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
  left_join(edx_user_romance_avgs, by=c('userId', 'Romance'))  %>%
  left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
  group_by(userId, SciFi)%>%
  summarize(b_u_s = sum((rating - mu - b_i-b_u - b_g - b_u_d- 
                           b_u_c - b_u_t- b_u_r - b_u_ch))/(n()+user_genre_effect_lambda))

edx_user_action_avgs <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>%
  left_join(edx_user_drama_avgs, by=c('userId', 'Drama'))  %>%
  left_join(edx_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
  left_join(edx_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
  left_join(edx_user_romance_avgs, by=c('userId', 'Romance'))  %>%
  left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
  left_join(edx_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
  group_by(userId, Action)%>%
  summarize(b_u_a = sum((rating - mu - b_i-b_u - b_g - b_u_d- 
                           b_u_c - b_u_t- b_u_r - b_u_ch-b_u_s))/(n()+user_genre_effect_lambda))



#check to see if the model is improved due to user-genre effect:

predicted_ratings <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>% 
  left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
  left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
  left_join(edx_user_thriller_avgs, by=c('userId','Thriller')) %>%
  left_join(edx_user_romance_avgs, by=c('userId','Romance')) %>%
  left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
  left_join(edx_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
  left_join(edx_user_action_avgs, by=c('userId', 'Action'))  %>%
  mutate(pred=mu+b_i + b_u + b_g + b_u_d + b_u_c 
         + b_u_t + b_u_r + b_u_ch + b_u_s + b_u_a)  %>%
  mutate(pred=ifelse(pred<.05, .05, pred)) %>%
  mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>%
  pull(pred)

rmse_user_genre<-RMSE(edx$rating, predicted_ratings)
print(paste("User genre rmse", rmse_user_genre))
print(paste("Model rmse improvement from user-genre component:", rmse_genre_effect-rmse_user_genre ))
gc()

print("Tuning parameter for movie-time effect")
cv_rmses<-sapply(lambda_seq, function(l) {
  print(paste("checking lambda", l))
  rmses<-sapply(k_seq, function(fold)
  {
    print(paste("fold", fold))
    cv_movies<- cv_movie_list[[fold]]
    mu_cv<-mu_cvs[fold]
    cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
    cv_user_avgs <- cv_user_avgs_list[[fold]]
    cv_genre_avgs <- cv_genre_avgs_list[[fold]]
    cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]] 
    cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]] 
    cv_user_thriller_avgs <-  cv_user_thriller_avgs_list[[fold]] 
    cv_user_romance_avgs <- cv_user_romance_avgs_list[[fold]] 
    cv_user_children_avgs <- cv_user_children_avgs_list[[fold]] 
    cv_user_scifi_avgs <- cv_user_scifi_avgs_list[[fold]]
    cv_user_action_avgs <- cv_user_action_avgs_list[[fold]]
    cv_movie_time_effect <- cv_movies %>% 
      left_join(cv_movie_avgs, by='movieId') %>%
      left_join(cv_user_avgs, by='userId') %>%
      left_join(cv_genre_avgs, by='genres')%>%
      left_join(cv_user_drama_avgs, by=c('userId','Drama')) %>%
      left_join(cv_user_comedy_avgs, by=c('userId','Comedy')) %>%
      left_join(cv_user_thriller_avgs, by=c('userId','Thriller')) %>%
      left_join(cv_user_romance_avgs, by=c('userId','Romance')) %>%
      left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
      left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
      left_join(cv_user_action_avgs, by=c('userId', 'Action'))  %>%
      group_by(movieId, year_rated) %>%
      summarize(b_m_y = sum(rating - mu_cv - b_i-b_u-b_g-b_u_d-b_u_c-
       b_u_t -b_u_r -b_u_ch - b_u_s-b_u_a) 
            /(n()+l))

    cv_predict <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
      left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
      left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
      left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
      left_join(cv_user_action_avgs, by=c('userId', 'Action'))  %>%
      left_join(cv_movie_time_effect, by=c('movieId', 'year_rated'))  %>%
      mutate(pred=mu_cv + b_i + b_u + b_g+ b_u_d+ b_u_c
             + b_u_t+b_u_r + b_u_ch + b_u_s +  b_m_y + b_u_a) %>% 
      mutate(pred=ifelse(pred<.05, .05, pred)) %>%
      mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>% pull(pred) 
    
    rmse<-RMSE(cv_movies$rating, cv_predict) 
    rmse
  }
  )
  mean(rmses)
 
} 
)
movie_time_effect_lambda<-lambda_seq[which.min(cv_rmses)]
print(paste("movie_time_effect_lambda", movie_time_effect_lambda))
print(paste("lowest movie time validation rmse", cv_rmses[which.min(cv_rmses)]))

#creating cv object list for movie time.

cv_movie_time_avgs_list<-lapply(k_seq, function(fold)
{
  
  mu_cv<-mu_cvs[fold]
  cv_movies<- cv_movie_list[[fold]]
  cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
  cv_user_avgs <- cv_user_avgs_list[[fold]]
  cv_genre_avgs <- cv_genre_avgs_list[[fold]]
  cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
  cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]]
  cv_user_thriller_avgs <- cv_user_thriller_avgs_list[[fold]]
  cv_user_romance_avgs <- cv_user_romance_avgs_list[[fold]]
  cv_user_children_avgs <- cv_user_children_avgs_list[[fold]]
  cv_user_scifi_avgs <- cv_user_scifi_avgs_list[[fold]]
  cv_user_action_avgs <- cv_user_action_avgs_list[[fold]]
  cv_movie_time_effect <- cv_movies %>% 
    left_join(cv_movie_avgs, by='movieId') %>%
    left_join(cv_user_avgs, by='userId') %>%
    left_join(cv_genre_avgs, by='genres')%>%
    left_join(cv_user_drama_avgs, by=c('userId','Drama')) %>%
    left_join(cv_user_comedy_avgs, by=c('userId','Comedy')) %>%
    left_join(cv_user_thriller_avgs, by=c('userId','Thriller')) %>%
    left_join(cv_user_romance_avgs, by=c('userId','Romance')) %>%
    left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
    left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
    left_join(cv_user_action_avgs, by=c('userId', 'Action'))  %>%
    group_by(movieId, year_rated) %>%
    summarize(b_m_y = sum(rating - mu_cv - b_i-b_u-b_g-b_u_d-b_u_c-
                            b_u_t -b_u_r -b_u_ch - b_u_s-b_u_a) /
              (n()+movie_time_effect_lambda)) %>%  mutate_at(c(3), ~replace(., is.na(.), 0))
  
} 
)

movie_time_effect <- edx %>% 
  left_join(edx_movie_avgs, by='movieId') %>%
  left_join(edx_user_avgs, by='userId') %>%
  left_join(edx_genre_avgs, by='genres')%>%
  left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
  left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
  left_join(edx_user_thriller_avgs, by=c('userId','Thriller')) %>%
  left_join(edx_user_romance_avgs, by=c('userId','Romance')) %>%
  left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
  left_join(edx_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
  left_join(edx_user_action_avgs, by=c('userId', 'Action'))  %>%
  group_by(movieId, year_rated) %>%
  summarize(b_m_y = sum((rating - mu - b_i-b_u-b_g-
                           b_u_d - b_u_c - b_u_t - b_u_r- b_u_ch- b_u_s-b_u_a))/(n()+movie_time_effect_lambda))


predicted_ratings <- edx %>%
  left_join(edx_movie_avgs, by='movieId') %>% 
  left_join(edx_user_avgs, by='userId') %>% 
  left_join(edx_genre_avgs, by='genres') %>% 
  left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
  left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
  left_join(edx_user_thriller_avgs, by=c('userId','Thriller')) %>%
  left_join(edx_user_romance_avgs, by=c('userId','Romance')) %>%
  left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
  left_join(edx_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
  left_join(edx_user_action_avgs, by=c('userId', 'Action'))  %>%
  left_join(movie_time_effect, by=c('movieId', 'year_rated')) %>%
  mutate(pred=mu+b_i + b_u + b_g + b_u_d + b_u_c 
         + b_u_t + b_u_r + b_u_ch + b_u_s + b_u_a+ b_m_y)  %>%
  mutate(pred=ifelse(pred<.05, .05, pred)) %>%
  mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>%
  pull(pred)


rmse_movie_time<-RMSE(edx$rating, predicted_ratings)
print(paste("Movie time rmse", rmse_movie_time))
print(paste("Model rmse improvement for movie_time component:", rmse_user_genre-rmse_movie_time ))



print("Tuning parameter for user-time effect")
cv_rmses<-sapply(lambda_seq, function(l) {
  print(paste("checking lambda", l))
  rmses<-sapply(k_seq, function(fold)
  {
    print(paste("fold", fold))
    cv_movies<- cv_movie_list[[fold]]
    mu_cv<-mu_cvs[fold]
    cv_movies<- cv_movie_list[[fold]]
    cv_movie_avgs<- cv_movie_avgs_list[[fold]] 
    cv_user_avgs <- cv_user_avgs_list[[fold]]
    cv_genre_avgs <- cv_genre_avgs_list[[fold]]
    cv_user_drama_avgs <- cv_user_drama_avgs_list[[fold]]
    cv_user_comedy_avgs <- cv_user_comedy_avgs_list[[fold]]
    cv_user_thriller_avgs <- cv_user_thriller_avgs_list[[fold]]
    cv_user_romance_avgs <- cv_user_romance_avgs_list[[fold]]
    cv_user_children_avgs <- cv_user_children_avgs_list[[fold]]
    cv_user_scifi_avgs <- cv_user_scifi_avgs_list[[fold]]
    cv_user_action_avgs <- cv_user_action_avgs_list[[fold]]
    cv_movie_time_effect <-cv_movie_time_avgs_list[[fold]]
   
    cv_user_time_effect <- cv_movies %>% 
      left_join(cv_movie_avgs, by='movieId') %>%
      left_join(cv_user_avgs, by='userId') %>%
      left_join(cv_genre_avgs, by='genres')%>%
      left_join(cv_user_drama_avgs, by=c('userId','Drama')) %>%
      left_join(cv_user_comedy_avgs, by=c('userId','Comedy')) %>%
      left_join(cv_user_thriller_avgs, by=c('userId','Thriller')) %>%
      left_join(cv_user_romance_avgs, by=c('userId','Romance')) %>%
      left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
      left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
      left_join(cv_user_action_avgs, by=c('userId', 'Action'))  %>%
      left_join(cv_movie_time_effect, by=c('movieId','year_rated')) %>%
          group_by(userId, year_rated) %>%
      summarize(b_u_y = sum((rating - mu_cv - b_i-b_u-b_g-
          b_u_d - b_u_c - b_u_t - b_u_r -b_u_ch - b_u_s -b_u_a- b_m_y))/(n()+l))
    
        
    cv_predict <- cv_movies %>%
      left_join(cv_movie_avgs, by='movieId') %>% 
      left_join(cv_user_avgs, by='userId') %>% 
      left_join(cv_genre_avgs, by='genres') %>%
      left_join(cv_user_drama_avgs, by=c('userId', 'Drama'))  %>%
      left_join(cv_user_comedy_avgs, by=c('userId', 'Comedy'))  %>%
      left_join(cv_user_thriller_avgs, by=c('userId', 'Thriller'))  %>%
      left_join(cv_user_romance_avgs, by=c('userId', 'Romance'))  %>%
      left_join(cv_user_children_avgs, by=c('userId', 'Children'))  %>%
      left_join(cv_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
      left_join(cv_user_action_avgs, by=c('userId', 'Action'))  %>%
          left_join(cv_movie_time_effect, by=c('movieId', 'year_rated'))  %>%
      left_join(cv_user_time_effect, by=c('userId', 'year_rated'))  %>%
      mutate(pred=mu_cv + b_i + b_u + b_g+ b_u_d+ b_u_c+ b_u_t+
               b_u_r + b_u_t + b_u_ch + b_u_s + b_u_a+b_m_y+ b_u_y) %>% 
      mutate(pred=ifelse(pred<.05, .05, pred)) %>%
      mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>% pull(pred) 
    
    rmse<-RMSE(cv_movies$rating, cv_predict) 
    rmse
  }
  )
  mean(rmses)
  
} 
)  
user_time_effect_lambda<-lambda_seq[which.min(cv_rmses)]
rmse_user_time<-lambda_seq[which.min(cv_rmses)]
print(paste("user time effect param", user_time_effect_lambda))
print(paste("user  time validation rmse", cv_rmses[which.min(cv_rmses)]))
gc()
# checking if user time effet improves the model

 user_time_effect <- edx %>% 
   left_join(edx_movie_avgs, by='movieId') %>%
   left_join(edx_user_avgs, by='userId') %>%
   left_join(edx_genre_avgs, by='genres')%>%
   left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
   left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
   left_join(edx_user_thriller_avgs, by=c('userId','Thriller')) %>%
   left_join(edx_user_romance_avgs, by=c('userId','Romance')) %>%
   left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
   left_join(edx_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
   left_join(edx_user_action_avgs, by=c('userId', 'Action'))  %>%
   left_join(movie_time_effect, by=c('movieId', 'year_rated')) %>%
   group_by(userId, year_rated) %>%
   summarize(b_u_y = sum((rating - mu - b_i-b_u-b_g -
        b_u_d - b_u_c - b_u_t - b_u_r - b_u_ch-b_u_s-b_u_a-b_m_y))/(n()+user_time_effect_lambda))

 predicted_ratings <- edx %>%
   left_join(edx_movie_avgs, by='movieId') %>% 
   left_join(edx_user_avgs, by='userId') %>% 
   left_join(edx_genre_avgs, by='genres') %>% 
   left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
   left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
   left_join(edx_user_thriller_avgs, by=c('userId','Thriller')) %>%
   left_join(edx_user_romance_avgs, by=c('userId','Romance')) %>%
   left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
   left_join(edx_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
   left_join(edx_user_action_avgs, by=c('userId', 'Action'))  %>%
   left_join(movie_time_effect, by=c('movieId', 'year_rated')) %>%
   left_join(user_time_effect, by=c('userId', 'year_rated'))%>%
   mutate(pred=mu+b_i + b_u + b_g + b_u_d + b_u_c 
          + b_u_t + b_u_r + b_u_ch + b_u_s + b_u_a + b_m_y + b_u_y)  %>%
   mutate(pred=ifelse(pred<.05, .05, pred)) %>%
   mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>%
   pull(pred)
 
 rmse_training<-RMSE(edx$rating, predicted_ratings)
print(paste("Training rmse", rmse_training))
print(paste("Model rmse improvement from user time component:", rmse_movie_time- rmse_training))
gc()
#apply model to validation/test set
 predicted_ratings <- validation %>%
   mutate(Drama=ifelse(str_detect(genres, 'Drama')==1, 1,0)) %>% 
   mutate(Comedy=ifelse(str_detect(genres, 'Comedy')==1, 1,0)) %>%  
   mutate(Thriller=ifelse(str_detect(genres, 'Thriller')==1, 1,0)) %>% 
   mutate(Romance=ifelse(str_detect(genres, 'Romance')==1, 1,0)) %>% 
   mutate(Children=ifelse(str_detect(genres, 'Children')==1, 1,0)) %>% 
   mutate(SciFi=ifelse(str_detect(genres, 'Sci-Fi')==1, 1,0)) %>% 
   mutate(Action=ifelse(str_detect(genres, 'Action')==1, 1,0)) %>% 
   mutate(year_rated = year(as_datetime(timestamp))) %>%
   left_join(edx_movie_avgs, by='movieId') %>% 
   left_join(edx_user_avgs, by='userId') %>% 
   left_join(edx_genre_avgs, by='genres') %>% 
   left_join(edx_user_drama_avgs, by=c('userId','Drama')) %>%
   left_join(edx_user_comedy_avgs, by=c('userId','Comedy')) %>%
   left_join(edx_user_thriller_avgs, by=c('userId','Thriller')) %>%
   left_join(edx_user_romance_avgs, by=c('userId','Romance')) %>%
   left_join(edx_user_children_avgs, by=c('userId', 'Children'))  %>%
   left_join(edx_user_scifi_avgs, by=c('userId', 'SciFi'))  %>%
   left_join(edx_user_action_avgs, by=c('userId', 'Action'))  %>%
   left_join(movie_time_effect, by=c('movieId', 'year_rated')) %>%
   left_join(user_time_effect, by=c('userId', 'year_rated'))%>%
   mutate(pred=mu+b_i + b_u + b_g + 
            ifelse(is.na(b_u_d), 0, b_u_d) + 
            ifelse(is.na(b_u_c), 0, b_u_c) + 
            ifelse(is.na(b_u_t), 0, b_u_t) + 
            ifelse(is.na(b_u_r), 0, b_u_r) + 
            ifelse(is.na(b_u_ch), 0, b_u_ch) + 
            ifelse(is.na(b_u_s), 0, b_u_s) + 
            ifelse(is.na(b_u_a), 0, b_u_a) + 
            ifelse(is.na(b_m_y), 0, b_m_y) + 
            ifelse(is.na(b_u_y), 0, b_u_y)
   ) %>%
   mutate(pred=ifelse(pred<.05, .05, pred)) %>%
   mutate(pred=ifelse(pred>5.0, 5.0, pred)) %>%
   pull(pred)
 rmse_test<- RMSE(validation$rating, predicted_ratings)

 print(paste("Test rmse", rmse_test))

 end_time<-Sys.time()
print(paste("model development run time in minutes:", difftime(end_time, start_time, units="mins")))
