setwd("C:/Users/Pichau/Desktop/Data _science/R/PROJETOS/projeto1")

#pacotes

library(ggplot2)
library(neuralnet)
library(caret)
library(dplyr)
library(randomForest)
library(data.table)


str(data)
#carregando dados

data <- fread("train_sample.csv", header = T)
set.seed(12)





## Verificando se existe algum erro com as variáveis temporais, vendo se existe registro de algum download ocorrendo antes do click na propaganda
x = data %>% filter(click_time - attributed_time > 0)
nrow(x)



#a coluna 'attributed_time' possui grande quantidade de valores NA, pois ela só está presente quando o download do app é relaizado. 
#checando a existência de valores NA para outras variáveis
sum(is.na(data[,-'attributed_time']))

#tirando as variáveis 'click_time' e 'attributed_time', todas as outras são fatores, porém parecem apresentar uma enorme quantidade de valores únicos.
#checando através de um plot a quantidade de valores únicos de cada uma das variáveis.
unique_values <- data[,-c('click_time','attributed_time')]

unique_values <- as.data.frame(apply(unique_values, 2, function(x) length(unique(x))))
unique_values$names <- rownames(unique_values)
colnames(unique_values) <- c('valores_unicos','variavel')
unique_values <- unique_values %>% arrange(desc(valores_unicos))

ggplot(unique_values, aes( x=reorder(variavel,-valores_unicos),valores_unicos)) +
  geom_bar(position='dodge', stat='identity',fill = "blue") + 
  geom_text(aes(label=valores_unicos), position=position_dodge(width=0.9), vjust=-0.25)

#podemos perceber uma enorme quantidade de IPs únicos. Importante notar que a variável que iremos prever possui apenas 2 valores únicos, que é o comportamento esperado.


# Em problemas de classificação, é importante que a variável a ser prevista esteja balanceada, ou seja, a quantidade de valores 0 e de valores 1 em 'is_attributed' devem ser parecidos.
# Vamos plotar essas quantidades

ggplot(data, aes(is_attributed)) + geom_bar(fill = 'grey3') +
  stat_count(geom = "text", colour = "white", size = 5,
             aes(label = ..count..),position=position_stack(vjust=0.5))

# Percebemos uma enorme necessidade de balancear os dados.Podemos fazer isso reduzindo nossa amostra onde 'is_attributed' for 0 ou criando dados em que 'is_attributed' é 1.
# Como temos acesso a uma quantidade muito grande de dados, usaremos uma técnica de downsample
yes <- which(data$is_attributed == 1)
no <- which(data$is_attributed == 0)
not_downloaded_sample <- sample(no, length(yes))
data = data[c(not_downloaded_sample,yes),]

ggplot(data, aes(is_attributed)) + geom_bar(fill = 'grey3') +
  stat_count(geom = "text", colour = "white", size = 5,
             aes(label = ..count..),position=position_stack(vjust=0.5))

#criando coluna com dia da semana e horário do clique
data$dayweek =  wday(data$click_time)
data$hour = hour(data$click_time)




###            anáise exploratória
##ANÁLISE - quem fez o download do app
#criando subset apenas com os dados dos que baixaram o app(data2)
data2 = subset(data, data$is_attributed == 1)

#análise: média de tempo levado para download do app
data2$delay = as.numeric(data2$attributed_time - data2$click_time)
mean(data2$delay/60)
sd(data2$delay/60)
hist(data2$delay, breaks = 10, col = 'lightblue')

#considerando pessoas que demoraram menos de 10min(600s)
delay_under10 = subset(data2, select = delay, subset = delay<600 )
mean(delay_under10$delay/60)
sd(delay_under10$delay/60)
hist(delay_under10$delay, breaks = 10, col = 'lightblue')

#% de pessoas que baixaram o app
a = nrow(data2) 
b = nrow(data)
a/b*100

#% de pessoas que baixaram o app em menos de 2 minutos(120s)
c = subset(data2, subset = data2$delay < 120)
d = nrow(c)
d/a*100

#conclusão: das pessoas que baixaram o app, 42,7% o fizeram em menos de 2 minutos,
#apesar da média ser de 74 minutos. Apenas 0.2% do total fizeram o download do app.
str(data)




# Descobrindo quantos dias estão presentesno nosso dataset
max(data$click_time) - min(data$click_time)


#Vamos criar um gráfico que mostra o horário do click em uma propaganda. Vamos comparar quem fez o download com quem não fez.

data %>%
  group_by(hour) %>%
  summarise(downloadsRealized = sum(is_attributed),
            unrealizedDownloads = sum(!is_attributed)) %>%
  ggplot(x = factor(hour)) +
  geom_line(aes(x = hour, y = downloadsRealized, color = 'Sim')) +
  geom_line(aes(x = hour, y = unrealizedDownloads, color = 'Não')) +
  theme_bw(base_size = 15) + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + 
  xlab('Hora') +
  ylab('Quantidade de cliques') +
  labs(title = 'Análise temporal de cliques', colour = 'Downloaded')


##Analizando os IPs que fizeram ou não o download.
# Ips que fizeram o download:
ip_downloaded = data %>% select(c('ip', 'is_attributed')) %>% filter(is_attributed == 1) %>% distinct()
#Ips que não fizeram o dowload
ip_not_donwloaded = data %>% select(c('ip', 'is_attributed')) %>% filter(is_attributed == 0) %>% distinct()
#IPs que pertencem aos dois grupos
ip_both = ip_downloaded %>% inner_join(ip_not_donwloaded , by = 'ip')

count(ip_downloaded)
count(ip_not_donwloaded)
count(ip_both)
# É possível perceber que apenas 2 IPs clicaram em uma propaganda e fez o download e clicou em outra e não o fez.
ip_only_downloaded = ip_downloaded %>% anti_join(ip_both)
ip_only_not_downloaded = ip_not_donwloaded %>% anti_join(ip_both)
total_IP = count(ip_downloaded) + count(ip_not_donwloaded) - count(ip_both)
count(ip_only_downloaded) / total_IP *100
count(ip_only_not_downloaded) / total_IP *100
#o total de Ips que nunca resultaram em cliques fraudulentos é de  % . Já o total que sempre resultou em clique fraudulento é de 
 



##            Data munging e Feature selection 



#eliminando colunas que não entrarão no modelo(variáveis temporais e IP, por possuir uma quantidade enorme de valores únicos)

data$attributed_time = NULL
data$click_time = NULL
data$ip = NULL


##análise exploratória
#comparativo: horário de click

ggplot(data, aes(hour)) +
  geom_bar(fill = 'deepskyblue3') + 
  facet_grid(. ~ is_attributed)

#comparativo: dia da semana
ggplot(data, aes(dayweek)) +
  geom_bar(fill = 'deepskyblue3') + 
  facet_grid(. ~ is_attributed)

#comparativo: app 
ggplot(data, aes(app)) +
  geom_histogram(fill = 'deepskyblue3') + 
  facet_grid(. ~ is_attributed)

#comparativo: device
ggplot(data, aes(device)) +
  geom_histogram(fill = 'deepskyblue3') + 
  facet_grid(. ~ is_attributed)

#comparativo: os
ggplot(data, aes(os)) +
  geom_bar(fill = 'deepskyblue3') + 
  facet_grid(. ~ is_attributed)

#comparativo: channel
ggplot(data, aes(channel)) +
  geom_bar(fill = 'deepskyblue3') + 
  facet_grid(. ~ is_attributed)



##Feature selection com ramdom forest

selection <- randomForest(is_attributed ~.,
                          data       = data, 
                          ntree = 100, nodesize = 10, importance = T)
varImpPlot(selection)
#variáveis fracas: day_week e hour
#criação dataset sem essas variáveis para testar no modelo(data_3)

data_3 = data
data_3$dayweek = NULL
data_3$hour =NULL


###   criação dos modelos

## Modelo 1 :Rede Neural(NN)

# como rede neural não aceita dados tipo fator, farei modificações nos atributos
data_nn <- data



#divisão dados treino e teste
sample <- sample.int(n = nrow(data_nn), size = floor(.7*nrow(data_nn)), replace = F)
train_sample_nn <- data_nn[sample, ]
test_sample_nn  <- data_nn[-sample, ]

#modelo
modelo_1 = neuralnet(is_attributed ~., train_sample_nn , hidden = 3)

#previsões
previsoes <- data.frame(observado = test_sample_nn$is_attributed,
                        previsto = predict(modelo_1, newdata = test_sample_nn))

previsoes$previsto <- ifelse (previsoes$previsto > 0.5, 1, 0)


#confusion matrix
confusionMatrix(table(previsoes$observado, previsoes$previsto))



##modelo 2: rede neural com menos parâmetros


data_nn_2 <- data_3


sample <- sample.int(n = nrow(data_nn_2), size = floor(.7*nrow(data_nn_2)), replace = F)
train_sample_nn_2 <- data_nn_2[sample, ]
test_sample_nn_2  <- data_nn_2[-sample, ]

#modelo
modelo_2 = neuralnet(is_attributed ~., train_sample_nn_2 , hidden = 3  )

#previsões
previsoes2 <- data.frame(observado = test_sample_nn_2$is_attributed,
                         previsto = predict(modelo_2, newdata = test_sample_nn_2))

previsoes2$previsto <- ifelse (previsoes2$previsto > 0.5, 1, 0)


#confusion matrix
confusionMatrix(table(previsoes2$observado, previsoes2$previsto))


##modelo 3 - ramdom forest

sample <- sample.int(n = nrow(data), size = floor(.7*nrow(data)), replace = F)
train_sample <- data[sample, ]
test_sample  <- data[-sample, ]

modelo_3 <- randomForest(is_attributed ~.,
                         data       = train_sample, 
                         ntree = 100, nodesize = 10)


previsoes3 <- data.frame(observado = test_sample$is_attributed,
                         previsto = predict(modelo_3, newdata = test_sample))

previsoes3$previsto <- ifelse (previsoes3$previsto > 0.5, 1, 0)

confusionMatrix(table(previsoes3$observado, previsoes3$previsto))

##modelo 4: ramdom forest com menos parâmetros

sample <- sample.int(n = nrow(data_3), size = floor(.7*nrow(data_3)), replace = F)
train_sample <- data_3[sample, ]
test_sample  <- data_3[-sample, ]

modelo_4 <- randomForest(is_attributed ~.,
                         data       = train_sample, 
                         ntree = 100, nodesize = 15)


previsoes4 <- data.frame(observado = test_sample$is_attributed,
                         previsto = predict(modelo_4, newdata = test_sample))

previsoes4$previsto <- ifelse (previsoes4$previsto > 0.5, 1, 0)
confusionMatrix(table(previsoes4$observado, previsoes4$previsto))



##conclusão: os modelos de ramdom forest apredentaram desempenho superior aos de redes neurais. 
#O modelo com menos parâmetros apresentou resultados semelhantes ao modelo com mais parâmetros,
#sendo preferível a sua utilização.








