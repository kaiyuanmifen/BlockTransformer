#organize data and plot 
library(ggplot2)

# 
# Block1=read.csv('block1.csv',header = F)
# head(Block1)
# Block1$Method="Block"
# Block1$seed=1
# 
# Block2=read.csv('block2.csv',header = F)
# head(Block2)
# Block2$Method="Block"
# Block2$seed=2
# 
# Block3=read.csv('block3.csv',header = F)
# Block3$Method="Block"
# head(Block3)
# Block3$seed=3
# 
# Vanilla1=read.csv('vanilla1.csv',header = F)
# head(Vanilla1)
# Vanilla1$Method="Vanilla"
# Vanilla1$seed=1
# 
# Vanilla2=read.csv('vanilla2.csv',header = F)
# 
# 
# Vanilla2$Method="Vanilla"
# Vanilla2$seed=2
# 
# Vanilla3=read.csv('vanilla3.csv',header = F)
# 
# Vanilla3$Method="Vanilla"
# Vanilla3$seed=3
# 
# 
# 
# Data=rbind(Block1,Block2,Block3,Vanilla1,Vanilla2,Vanilla3)
# head(Data)

#Data=Data[,c(3,8,12,16,17,18)]
#names(Data)=c("Episode","Ternary_accuracy","Relations_accuracy","Non_relations_accuracy","Method","seed")

AllDirs=list.files("./")
AllDirs=AllDirs[grepl(AllDirs,pattern = "Exp")]


Data=NULL
for( DIR in AllDirs){

  vec=read.csv(paste0(DIR,"/",list.files(DIR)[1]),header = F)
  vec$Method=strsplit(DIR,split = "-")[[1]][4]
  
  Data=rbind(Data,vec)
  
}
head(Data)
names(Data)=c("epoch", "train_acc_ternary", 
              "train_acc_binary",
              "train_acc_unary", 
              "test_acc_ternary", 
              "test_acc_binary", 
              "test_acc_unary","Method")

# 
# Data$Ternary_accuracy=as.numeric(sub("%", "", Data$Ternary_accuracy))
# Data$Relations_accuracy=as.numeric(sub("%", "", Data$Relations_accuracy))
# Data$Non_relations_accuracy=as.numeric(sub("%", "", Data$Non_relations_accuracy))
# 
# Data=Data[!is.na(Data$Ternary_accuracy),]
# unique(Data$Ternary_accuracy)
# Data$Episode=as.numeric(Data$Episode)
head(Data)





ggplot(data = Data, aes(x=epoch, y=test_acc_ternary)) +
  geom_smooth(size=2)


ggplot(data = Data, aes(x=epoch, y= test_acc_unary,linetype=Method,colour=Method)) +
  geom_smooth(size=2)

ggplot(data = Data, aes(x=epoch, y=test_acc_ternary,linetype=Method,colour=Method)) +
  geom_smooth(size=2)+xlim(0,20)


ggplot(data = Data, aes(x=epoch, y=test_acc_binary,linetype=Method,colour=Method)) +
  geom_smooth(size=2)


ggplot(data = Data, aes(x=epoch, y=test_acc_unary,linetype=Method,colour=Method)) +
  geom_smooth(size=2)




ggplot(data = Data, aes(x=Episode, y=Non_relations_accuracy,linetype=Method,colour=Method)) +
  geom_smooth(size=2)



ggplot(data = Data, aes(x=Episode, y= ppl,linetype=Method,colour=Type)) +
  geom_line(size=1)


ggplot(data = Data, aes(x=Episode, y= accuracy ,linetype=Method,colour=Type)) +
  geom_line(size=1)



ggplot(data = VecPlot, aes(x=Episode, y=trainReward,linetype=KL_coefficient,colour=KL_coefficient)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")+ggtitle(Method)



ggplot(data = VecPlot, aes(x=Episode, y=OODtestReward,linetype=KL_coefficient,colour=KL_coefficient)) +
  geom_smooth(size=2)+scale_color_brewer(palette="Paired")+ggtitle(Method)



VecPlot=Data[(!is.na(Data$Episode))&(Data$Episode>=30)&(Data$N_agents==N_agent)&(!is.na(Data$KL_coefficient))&(Data$Method==Method),]

VecPlot$KL_coefficient=as.factor(VecPlot$KL_coefficient)
ggplot(VecPlot, aes(x=KL_coefficient, y=trainReward,color=KL_coefficient)) + 
  geom_boxplot()+ coord_flip()+
  ggtitle(paste("Done env, N_agent=",N_agent,Method))



ggplot(VecPlot, aes(x=KL_coefficient, y=OODtestReward,color=KL_coefficient)) + 
  geom_boxplot()+ coord_flip()+
  ggtitle(paste("Done env, N_agent=",N_agent,Method))






VecPlot=Data[(!is.na(Data$Episode))&(Data$Episode>40)&(Data$N_agents==N_agent),]


ggplot(VecPlot, aes(x=Method, y=OODtestReward,color=Method)) + 
  geom_boxplot()+ coord_flip()+
    ggtitle(paste("Done env, N_agent=",N_agent))

ggsave(filename = paste0("../Images/Drone/OODTest_",N_agent,".png"),height = 7,width = 11)



ggplot(VecPlot, aes(x=Method, y=testReward,color=Method)) + 
  geom_boxplot()+ coord_flip()+
  ggtitle(paste("Done env, N_agent=",N_agent))

ggsave(filename = paste0("../Images/Drone/IndistributionTest_",N_agent,".png"),height = 7,width = 11)



