
rm(list=ls())
options(digits=5) 
#load required packages.
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tibble)) install.packages("tibble", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(pls)) install.packages("pls", repos = "http://cran.us.r-project.org")
options(dplyr.summarise.inform=F)
library(tidyverse)
library(caret)
library(data.table)
library(tibble)
library(matrixStats)
library(dplyr)
library(ggplot2)
library(knitr)
library(pls)


set.seed(1)
#function used to assess errors in prediction
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#load data.
nba_player_stats<-read.csv("https://raw.githubusercontent.com/czimmerman19/RProjects/master/NBAMoneyBall/CSV/NBA_Players_Stats 201819.csv")
nba_team_stats<-read.csv("https://raw.githubusercontent.com/czimmerman19/RProjects/master/NBAMoneyBall/CSV/nba_team_stats_00_to_18.csv")
nba_player_salaries<-read.csv("https://raw.githubusercontent.com/czimmerman19/RProjects/master/NBAMoneyBall/CSV/NBA_Player_Salaries_201809.csv")
#player data looks like this:
#Bam Adebayo\\adebaba01
#we want to create two columns, Player Name and player id
lst_Players<-str_split(nba_player_stats %>% pull(Player), "\\\\")

# lst_Players=list with a value pair, e.g.,  "Quincy Acy" "acyqu01"
p<-unlist(lst_Players)
#turns into a single vector doublet the size, e.g,. 
#p[1:2]
#"Quincy Acy" 
#"acyqu01"

#split above into player id and player name based on position
seq_name<-seq(1,nrow(nba_player_stats)*2, 2)
seq_id<-seq(2,nrow(nba_player_stats)*2, 2)

#get rid of meaningless column
nba_player_stats <- nba_player_stats  %>%  select(-Rk)

#remove Player and add back Player + Player Id
nba_player_stats <- nba_player_stats  %>%  select(-Player)
nba_player_stats <- nba_player_stats%>% cbind(Player=p[seq_name], PlayerId=p[seq_id])

#move player name to the left to make dataframe easier to read
num_col<-ncol(nba_player_stats)
nba_player_stats <- nba_player_stats[, c(num_col-1 , num_col,
      1:(num_col-2))]


#get the players in both data sets
nba_player_stats_salaries <-
  nba_player_stats %>% inner_join(nba_player_salaries, by=c("Player"))
  
  #now we need to make this unique on player, there are players with multiple rows
  #because they played on multiple teams.  
  #we need to make it unique on PlayerId and retain the row where Team='TOT'
  dups<-nba_player_stats_salaries %>% group_by(PlayerId) %>%
  summarize(n=n()) %>% filter(n>1)
dup_rows_to_keep <-nba_player_stats_salaries %>% 
  inner_join(dups, by=c("PlayerId")) %>% select(-n) %>%
   filter(Tm=='TOT')

nba_player_stats_salaries <- nba_player_stats_salaries %>%
  anti_join(dups, by=c("PlayerId"))
nba_player_stats_salaries <-
    rbind(nba_player_stats_salaries,dup_rows_to_keep)
# set is.na to 0 (e.g., players with 0 3pt attempts)
nba_player_stats_salaries[is.na(nba_player_stats_salaries)] <- 0
#confirm no dups
nba_player_stats_salaries %>% group_by(PlayerId) %>%
  summarize(n=n()) %>% filter(n>1)



#turn the player data frame stats into team equivalents by adjusting stats 
#as if they played the whole game and multiplying by 5 
#also limit to player with at least 15 minutes per game.
avg_mins_game<-mean(nba_team_stats$MIN)

#limit to player with at least 15 minutes per game., 
#25 GP and salary > minimum in 2018-2019 for rookies in of 838,464 (less)
#would mean a partial season contract.


 nba_min_salary_2018_2019<-838464
       
nba_player_stats_salaries_adj<-nba_player_stats_salaries %>% 
    filter(MP>=15 & Salary >nba_min_salary_2018_2019 & G>25) %>%
  mutate(PTS=PTS*(avg_mins_game/MP)*5,FGM=FGM*(avg_mins_game/MP)*5,
         FGA=FGA*(avg_mins_game/MP)*5,ThreePM=ThreePM*(avg_mins_game/MP)*5,
         ThreePA=ThreePA*(avg_mins_game/MP)*5, FTM=FTM*(avg_mins_game/MP)*5,
         FTA=FTA*(avg_mins_game/MP)*5, ORB=ORB*(avg_mins_game/MP)*5,
         DRB=DRB*(avg_mins_game/MP)*5,TRB=TRB*(avg_mins_game/MP)*5,
         AST=AST*(avg_mins_game/MP)*5,TOV=TOV*(avg_mins_game/MP)*5,
         STL=STL*(avg_mins_game/MP)*5,BLK=BLK*(avg_mins_game/MP)*5,
         PF=PF*(avg_mins_game/MP)*5)

#now scale all columns that will be used in analysis so that metrics are equivalent
nba_team_stats_adj <-nba_team_stats %>% mutate(WPct=as.numeric(scale(WPct)), PTS=as.numeric(scale(PTS)),
    FGM=as.numeric(scale(FGM)), FGA=as.numeric(scale(FGA)), FGPct=as.numeric(scale(FGPct)), 
    ThreePM=as.numeric(scale(ThreePM)),
    ThreePA=as.numeric(scale(ThreePA)), ThreeP_Pct=as.numeric(scale(ThreeP_Pct)), 
    FTM=as.numeric(scale(FTM)),
    FTA=as.numeric(scale(FTA)), FT_Percent=as.numeric(scale(FT_Percent)), 
    ORB=as.numeric(scale(ORB)), DRB=as.numeric(scale(DRB)),
    TRB=as.numeric(scale(TRB)), AST=as.numeric(scale(AST)), 
    TOV=as.numeric(scale(TOV)), STL=as.numeric(scale(STL)), 
    BLK=as.numeric(scale(BLK)),    PF=as.numeric(scale(PF)))
    #same for the player stats
nba_player_stats_salaries_adj <-nba_player_stats_salaries_adj    %>%
mutate( PTS=as.numeric(scale(PTS)),
        FGM=as.numeric(scale(FGM)), FGA=as.numeric(scale(FGA)), FGPct=as.numeric(scale(FGPct)), 
        ThreePM=as.numeric(scale(ThreePM)),
        ThreePA=as.numeric(scale(ThreePA)), ThreeP_Pct=as.numeric(scale(ThreeP_Pct)), 
        FTM=as.numeric(scale(FTM)),
        FTA=as.numeric(scale(FTA)), FT_Percent=as.numeric(scale(FT_Percent)), 
        ORB=as.numeric(scale(ORB)), DRB=as.numeric(scale(DRB)),
        TRB=as.numeric(scale(TRB)), AST=as.numeric(scale(AST)), 
        TOV=as.numeric(scale(TOV)), STL=as.numeric(scale(STL)), 
        BLK=as.numeric(scale(BLK)),    PF=as.numeric(scale(PF)))                                

#create version of team data for modeling.
nba_team_stats_for_modeling <- nba_team_stats_adj %>%
  select (-TEAM, -GP, -W, -L, -MIN,  - SEASON, -BLKA,-Plus_Minus, -PFD) 
 
#end of data set up, begin model development.
#model 1: step wise regresion.


train.control <- trainControl(method = "cv", number = 5)
# Train the model
step_fit <- train(WPct ~., data = nba_team_stats_for_modeling,
                    method = "leapSeq", 
                    trControl = train.control)


Y_hat_step <- predict(step_fit, nba_team_stats_for_modeling)
step_rmse<-RMSE( nba_team_stats_for_modeling$WPct, Y_hat_step)
#uncomment below to view chosen vars for the model (FGM, FGA, TRB, TOV)
#summary(step_fit$finalModel) 

Y_hat_step_fit<-predict(step_fit,nba_player_stats_salaries_adj)

step_player_ranking<-nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=Y_hat_step_fit)%>%  
  arrange(desc(WinPctAsTeam)) %>% mutate(rank=row_number()) %>% 
  select(Player, WinPctAsTeam,rank) %>%
  filter(Player %in% c("James Harden", "Stephen Curry",
      "Giannis Antetokounmpo","LeBron James")) %>% summarize(ranking=sum(rank)) %>% 
  ungroup() %>%
  pull(ranking)

#model 2: simple multiple regression
#uncomment below to see correlations for outcome and all predictors, team data set
# correlation_matrix<-nba_team_stats_adj %>% 
#    select (WPct,  AST ,FGPct,ThreeP_Pct,DRB) %>% cor()
#  correlation_matrix

#uncomment below to see correlations for all predictors, player data set
 # plyr_correlation_matrix<-nba_player_stats_salaries %>% 
 #   select(AST ,FGPct,ThreeP_Pct,DRB) %>% cor()
 # plyr_correlation_matrix
 
 #train the model
theory_fit <- train(WPct ~ FGPct+ThreeP_Pct+DRB+AST ,
                    data = nba_team_stats_for_modeling,
                 method="lm",
                    trControl = train.control)
#predict winning pct for team
Y_hat_theory_fit <- predict(theory_fit,nba_team_stats_for_modeling)
#check rmse
theory_rmse<-RMSE(nba_team_stats_for_modeling$WPct, Y_hat_theory_fit)
#predict player stats
Y_hat_player_theory_fit <-predict(theory_fit,nba_player_stats_salaries_adj)
#generate ranking statistic used to validate model
theory_player_ranking<-nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=Y_hat_player_theory_fit)%>%  
  arrange(desc(WinPctAsTeam)) %>% mutate(rank=row_number()) %>% 
  select(Player, WinPctAsTeam,rank)%>%
  filter(Player %in% c("James Harden", "Stephen Curry",
                       "Giannis Antetokounmpo","LeBron James")) %>% 
  summarize(ranking=sum(rank))  %>%  ungroup() %>%
  pull(ranking)
#uncomemnt below to see predictors with significant p value (FGPCT,ThreeP_Pct, DRB )
# tidy(theory_fit$finalModel) %>% filter(p.value<=.05) %>% arrange(p.value)


#Partial least squares regression
#fit plrs model
pls.model =plsr(WPct ~ .,data=nba_team_stats_for_modeling,validation ="CV")

#2 components
pls_pred_2_comp <- predict(pls.model, nba_team_stats_adj, ncomp=2)
pls_2comp_rmse<-RMSE(nba_team_stats_for_modeling$WPct, as.numeric(pls_pred_2_comp))
Y_hat_player_pls_twocomp<-predict(pls.model,nba_player_stats_salaries_adj,ncomp = 2)
pls_2comp_player_ranking<-nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=as.numeric(Y_hat_player_pls_twocomp)) %>%  
  arrange(desc(WinPctAsTeam)) %>% mutate(rank=row_number()) %>% 
  select(Player, WinPctAsTeam,rank)%>%
  filter(Player %in% c("James Harden", "Stephen Curry",
                       "Giannis Antetokounmpo","LeBron James")) %>% 
  summarize(ranking=sum(rank))%>%  ungroup() %>%
  pull(ranking)


#1 component.
pls_pred_1_comp <- predict(pls.model, nba_team_stats_adj, ncomp=1)
pls_1comp_rmse<-RMSE(nba_team_stats_for_modeling$WPct, as.numeric(pls_pred_1_comp))


Y_hat_player_pls_onecomp<-
  Y_hat_player_pls_onecomp<-predict(pls.model,nba_player_stats_salaries_adj,ncomp = 1)

pls_1comp_player_ranking<-nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=as.numeric(Y_hat_player_pls_onecomp)) %>%  
  arrange(desc(WinPctAsTeam)) %>% mutate(rank=row_number()) %>% 
  select(Player, WinPctAsTeam,rank)%>%
  filter(Player %in% c("James Harden", "Stephen Curry",
                       "Giannis Antetokounmpo","LeBron James")) %>% 
  summarize(ranking=sum(rank))%>%  ungroup() %>%
  pull(ranking)


#uncomemnt below to view most important model components
# data.frame(pls.model$projection) %>% select(Comp.1) %>% mutate(loading=abs(Comp.1)) %>%
#   cbind(var=rownames(pls.model$projection)) %>% arrange(desc(loading))


#Begin lasso and ridge regression. Requires matrix as x variable
x <- model.matrix(WPct~., nba_team_stats_for_modeling)[,-1]
# Outcome variable
y <- nba_team_stats_for_modeling$WPct

#this will use the optimal lambda parameter, it applies regularization penalty:
#alpha: :
# “1”: for lasso regression
#“0”: for ridge regression

cv <- cv.glmnet(x, y, alpha = 0,dfmax=5) #ridge
cv_ridge<-cv$lambda.min
cv <- cv.glmnet(x, y, alpha = 1,dfmax=5)#lasso
cv_lasso<-cv$lambda.min

#train ridge model
ridge_fit<-glmnet(x, y, alpha = 0,  lambda = cv_ridge)
ridge_pred<- ridge_fit %>% predict(x) %>% as.vector()
# obtain rmse, ridge
ridge_rmse<-RMSE(nba_team_stats_adj$WPct,ridge_pred )

#need same predictor matrix for players
x_player <-
  nba_player_stats_salaries_adj %>% select("PTS","FGM", "FGA",        
  "FGPct","ThreePM","ThreePA","ThreeP_Pct","FTM", "FTA",  "FT_Percent","ORB","DRB",
  "TRB",  "AST","TOV", "STL", "BLK", "PF") %>% as.matrix()
Y_hat_player_ridge<-predict(ridge_fit ,x_player)

#obtain ranking stat, ridge
ridge_player_ranking<-nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=as.numeric(Y_hat_player_ridge)) %>%  
  arrange(desc(WinPctAsTeam)) %>% mutate(rank=row_number()) %>% 
  select(Player, WinPctAsTeam,rank)%>%
  filter(Player %in% c("James Harden", "Stephen Curry",
                       "Giannis Antetokounmpo","LeBron James")) %>% 
  summarize(ranking=sum(rank))%>%  ungroup() %>%
  pull(ranking)


#Lasso regression

lasso_fit<-glmnet(x, y, alpha = 1, lambda = cv_lasso)
lasso_pred<- lasso_fit %>% predict(x) %>% as.vector()
#obtain rmse lasso
lasso_rmse<-RMSE(nba_team_stats_adj$WPct,lasso_pred )

Y_hat_player_lasso<-predict(lasso_fit,x_player)

#obtain ranking stat, lasso
lasso_player_ranking<-nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=as.numeric(Y_hat_player_lasso)) %>%  
  arrange(desc(WinPctAsTeam)) %>% mutate(rank=row_number()) %>% 
  select(Player, WinPctAsTeam,rank)%>%
  filter(Player %in% c("James Harden", "Stephen Curry",
                       "Giannis Antetokounmpo","LeBron James")) %>% 
  summarize(ranking=sum(rank))%>%  ungroup() %>%
  pull(ranking)


#below was used to know values to use for position to select money ball team
#nba_player_stats_salaries_adj %>% select(Pos) %>% distinct() %>%  pull(Pos)
# "C"     "PF"    "SF"    "PG"    "SG"    "PF-SF" "SF-SG" "SG-PF" "C-PF"  "SG-SF"
  
#average NBA salary for season of interest.
  avg_salary_2018_2019<-6388007

  #select a point guard
 MoneyBallTeam<- nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=as.numeric(Y_hat_player_pls_onecomp)) %>% 
  mutate(worth=WinPctAsTeam/as.numeric(Salary))  %>% arrange(desc(worth)) %>%
  filter(Pos  %in% c("PG")& (Salary<  avg_salary_2018_2019)) %>% slice(1) %>% pull(Player)
 
#select a shooting guard
MoneyBallTeam<-append(MoneyBallTeam, nba_player_stats_salaries_adj %>% 
  cbind(WinPctAsTeam=as.numeric(Y_hat_player_pls_onecomp)) %>% 
  mutate(worth=WinPctAsTeam/as.numeric(Salary)) %>% arrange(desc(worth)) %>%
  filter(Pos  %in% c("SG", "SF-SG", "SG-PF", "SG-SF") & (Salary<avg_salary_2018_2019) 
         & !Player %in% c(MoneyBallTeam))%>% 
slice(1) %>% pull(Player))

#select a small forward 
MoneyBallTeam<-append(MoneyBallTeam, nba_player_stats_salaries_adj %>% 
                cbind(WinPctAsTeam=as.numeric(Y_hat_player_pls_onecomp)) %>% 
                mutate(worth=WinPctAsTeam/as.numeric(Salary)) %>% arrange(desc(worth)) %>%
                filter(Pos  %in% c("SF", "PF-SF", "SF-SG", "SG-SF") &  (Salary< avg_salary_2018_2019) &
                         !Player 
                       %in% c(MoneyBallTeam)) %>% slice(1) %>% pull(Player))
#select a power forward 
MoneyBallTeam<-append(MoneyBallTeam, nba_player_stats_salaries_adj %>% 
                        cbind(WinPctAsTeam=as.numeric(Y_hat_player_pls_onecomp)) %>% 
                        mutate(worth=WinPctAsTeam/as.numeric(Salary)) %>% arrange(desc(worth)) %>%
                        filter(Pos  %in% c("PF", "PF-SF", "SG-PF", "C-PF") &  (Salary<avg_salary_2018_2019) &
                                 !Player %in% c(MoneyBallTeam))  %>%
                      slice(1) %>% pull(Player))
#select a center
MoneyBallTeam<-append(MoneyBallTeam, nba_player_stats_salaries_adj %>% 
                        cbind(WinPctAsTeam=as.numeric(Y_hat_player_pls_onecomp)) %>% 
                        mutate(worth=WinPctAsTeam/as.numeric(Salary)) %>% arrange(desc(worth)) %>%
                        filter(Pos  %in% c("C", "C-PF")  &  (Salary<avg_salary_2018_2019) & !Player %in% c(MoneyBallTeam))  %>%
                        slice(1) %>% pull(Player))

names(MoneyBallTeam)<-c("PG", "SG", "SF", "PF","C")

#display money ball tea,
MoneyBallTeam

#uncomment to compare moneyball team salary to average for 5 players
# sum(nba_player_stats_salaries_adj %>% filter(Player %in% MoneyBallTeam) %>% pull(Salary))
# nba_player_stats_salaries_adj %>% filter(Player %in% MoneyBallTeam) %>% 
#   select(Player, Salary)
# avg_salary_2018_2019 * 5

