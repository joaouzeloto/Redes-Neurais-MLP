# 🧠 Implementação de Rede Neural MLP com Backpropagation

## 📚 Trabalho de Inteligência Artificial - 2º Bimestre

Este repositório contém a implementação de uma ferramenta para o **treinamento de uma Rede Neural Multilayer Perceptron (MLP)** utilizando o algoritmo **Backpropagation**, desenvolvida como parte do 2º trabalho da disciplina de Inteligência Artificial.

---

## 📌 Descrição do Projeto

O projeto consiste na criação de uma rede neural capaz de aprender a partir de um conjunto de dados de entrada, fornecido por arquivos `.csv`, com o objetivo de realizar **classificação supervisionada** em um conjunto de 5 classes de saída.


---

## ⚙️ Funcionalidades

- Leitura de arquivos `.csv` para treinamento e teste;
- Definição do **número de neurônios na camada oculta** (com sugestões baseadas na média aritmética ou geométrica);
- Escolha da **função de ativação** para os neurônios:
  - Linear
  - Logística (Sigmoid)
  - Tangente Hiperbólica (Tanh)
- Definição da **taxa de aprendizagem (η)** entre 0 e 1;
- Escolha do **critério de parada**:
  - Por número máximo de épocas (iterações)
  - Por erro mínimo atingido
- **Geração da Matriz de Confusão** para avaliação de desempenho da rede após o teste.

---

## 🧮 Fórmulas Importantes

### Erro na Camada de Saída:

- **Sigmoid**:  
  `ErroG = (Desejado - iG) * (iG * (1 - iG))`

- **Tangente Hiperbólica**:  
  `ErroG = (Desejado - iG) * (1 - iG²)`

### Erro na Camada Oculta:

- **Sigmoid**:  
  `ErroC = (ErroG * peso) * (iC * (1 - iC))`

- **Tangente Hiperbólica**:  
  `ErroC = (ErroG * peso) * (1 - iC²)`



