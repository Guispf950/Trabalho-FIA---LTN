# Projeto 3: Raciocínio Espacial Neuro-Simbólico com Logic Tensor Networks (LTN)

**Disciplina:** Fundamentos de Inteligência Artificial (FIA)

**Professor:** Edjard Mota

## 👥 Equipe
- ANDRÉ MALMSTEEN OLIVEIRA AMORIM
- BENJAMIM ISAAC RIBEIRO LIMA
- DIEGO GABRIEL SILVA AZEVEDO
- GUILHERME DA SILVA PEREIRA
- LETÍCIA ARAÚJO
- MANFRED LIMA VEIGA

---

# 🧠 Raciocínio Espacial Neuro-Simbólico com Logic Tensor Networks (LTN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![LTN](https://img.shields.io/badge/AI-Neuro--Symbolic-green)
![Status](https://img.shields.io/badge/Status-Concluído-success)

Este projeto implementa um agente **Neuro-Simbólico** capaz de aprender e raciocinar sobre relações espaciais (esquerda, direita, abaixo, empilhamento) em um ambiente 2D simplificado.

Diferente de redes neurais tradicionais "caixa-preta", este modelo utiliza **Logic Tensor Networks (LTN)** para aprender conceitos guiados por uma base de conhecimento rigorosa composta por **20 axiomas lógicos**.

---

## 📑 Sumário
1. [Visão Geral e Objetivo](#-visão-geral-e-objetivo)
2. [Estrutura dos Dados (CLEVR-Simplified)](#-estrutura-dos-dados)
3. [Predicados (O Vocabulário do Modelo)](#-predicados-o-vocabulário-do-modelo)
4. [Os 20 Axiomas Lógicos (As Regras do Jogo)](#-os-20-axiomas-lógicos-as-regras-do-jogo)
5. [Metodologia de Treino e Teste](#-metodologia-de-treino-e-teste)
6. [Análise Aprofundada dos Resultados](#-análise-aprofundada-dos-resultados)
7. [Como Executar](#-como-executar)

---

## 🔭 Visão Geral e Objetivo

O objetivo deste trabalho é integrar o aprendizado profundo (Deep Learning) com a lógica formal. O sistema deve:
1.  Receber vetores numéricos representando objetos geométricos.
2.  Aprender o significado de conceitos espaciais (`LeftOf`, `Below`, etc.) sem rótulos supervisionados diretos, apenas satisfazendo regras lógicas.
3.  Responder a perguntas complexas (Queries) sobre o cenário.

---

## 💾 Estrutura dos Dados

O ambiente simula objetos geométricos com vetores de **11 dimensões**:
* `[0, 1]`: Coordenadas X, Y (normalizadas 0.0 a 1.0).
* `[2, 3, 4]`: One-hot vector para cores (**Vermelho, Verde, Azul**).
* `[5, 6, 7, 8, 9]`: One-hot vector para formas (**Círculo, Quadrado, Cilindro, Cone, Triângulo**).
* `[10]`: Tamanho (Contínuo: 0.0 a 1.0).

---

## 🗣 Predicados (O Vocabulário do Modelo)

Os predicados são as "palavras" que a IA usa para descrever o mundo. Eles são mapeados para redes neurais (MLP) ou funções lógicas.

### Predicados Unários (Atributos)
Verificam propriedades de um único objeto ($P(x) \rightarrow [0,1]$):
* **Formas:** `IsCircle(x)`, `IsSquare(x)`, `IsCylinder(x)`, `IsCone(x)`, `IsTriangle(x)`.
* **Cores:** `IsRed(x)`, `IsGreen(x)`, `IsBlue(x)`.
* **Tamanho:** `IsSmall(x)`, `IsLarge(x)`.

### Predicados Binários (Relações)
Verificam a relação entre dois objetos ($R(x,y) \rightarrow [0,1]$):
* **Espaciais Horizontais:** `LeftOf(x,y)`, `RightOf(x,y)`.
* **Espaciais Verticais:** `Below(x,y)` (Abaixo), `Above(x,y)` (Acima).
* **Físicos/Outros:**
    * `CloseTo(x,y)`: Baseado na distância Euclidiana (Gaussiana).
    * `SameSize(x,y)`: Verifica similaridade de tamanho.
    * `CanStack(x,y)`: Verifica se $x$ pode ser empilhado sobre $y$.

### Predicados Ternários
* **Posicional:** `InBetween(x, y, z)`: Verifica se o objeto $x$ está espacialmente entre $y$ e $z$.

---

## 📜 Os 20 Axiomas Lógicos (As Regras do Jogo)

O coração do sistema. O modelo é treinado para maximizar a verdade destas 20 regras simultaneamente.

### 🔹 Grupo 1: Taxonomia e Física Básica
1.  **Exclusividade de Forma:** Cones não podem ser Quadrados. ($\forall x, Cone(x) \rightarrow \neg Square(x)$).
2.  **Tamanho de Forma:** Todo Cone é considerado Grande.
3.  **Restrição de Cor:** Círculos não podem ser Vermelhos.
4.  **Semântica de Cor:** Objetos Vermelhos e Verdes nunca estão Próximos (`CloseTo`).
5.  **Tamanho Disjuntivo:** Triângulos são ou Pequenos ou Grandes.

### 🔹 Grupo 2: Relações Espaciais (Horizontal/Vertical)
6.  **Existencial Vertical:** Todo Quadrado Azul tem algum Verde abaixo dele.
7.  **Existencial Horizontal:** Todo Quadrado tem algo à sua direita (está à esquerda de alguém).
8.  **Restrição de Posição:** Se um objeto está `InBetween` (entre outros), ele não pode ser um Triângulo.
9.  **Definição de InBetween:** Estar entre $y$ e $z$ significa estar à esquerda de um e à direita do outro.
10. **Inversa:** `LeftOf(x,y)` é equivalente a `RightOf(y,x)`.

### 🔹 Grupo 3: Queries Complexas (O Desafio)
11. **Query Q1 (Composta):** Existe objeto Pequeno que está Abaixo de um Cilindro E à Esquerda de um Quadrado?
12. **Query Q2 (Absoluta):** Existe um Cone Verde entre dois objetos quaisquer?
13. **Query Q3 (Regra Aprendida):** Se dois triângulos estão próximos, eles *devem* ter o mesmo tamanho.

### 🔹 Grupo 4: Axiomas Estruturais (Rigor Lógico)
   Para garantir que a IA não "alucine" relações impossíveis:

14. **Irreflexividade:** Nada está à esquerda de si mesmo `¬LeftOf(x,x)`.
15. **Assimetria Horizontal:** Se x está à esquerda de y, y **não** pode estar à esquerda de x.
16. **Transitividade Horizontal:** Se x < y e y < z, então x < z.
17. **Transitividade Vertical:** Se x está abaixo de y e y abaixo de z, então x abaixo de z.

### 🔹 Grupo 5: Definições Avançadas
18. **LastOnTheLeft:** Define o conceito de "objeto mais à esquerda de todos".
19. **LastOnTheRight:** Define o conceito de "objeto mais à direita de todos".
20. **CanStack (Empilhamento):** Define que $x$ pode empilhar em $y$ somente se a base $y$ for estável (Quadrado/Cilindro) e houver equilíbrio.

---

## 🔬 Metodologia de Treino e Teste

Para cumprir os requisitos pedagógicos da disciplina:

1.  **Treino Estático (Static Scene):** O modelo treina em um único cenário fixo (25 objetos com posições imutáveis). Isso força a rede a aprender as *regras* lógicas abstratas, já que não há variação de dados para memorizar estatisticamente.
2.  **Teste Aleatório (Random Scenes):** O modelo treinado é avaliado em 5 cenários gerados totalmente ao acaso. O sucesso aqui prova a **generalização**.

---

## 📊 Análise Aprofundada dos Resultados

O experimento foi conduzido sob uma metodologia rigorosa para garantir que a rede neural não apenas memorizasse o cenário, mas aprendesse a **lógica espacial subjacente**.

* **Treinamento:** 1 Cenário Estático (Overfitting forçado em regras).
* **Teste:** 5 Cenários Aleatórios Independentes (Teste de Generalização).

Abaixo, apresentamos a média consolidada das 5 execuções e a interpretação fenomenológica dos dados.

### 1. Tabela Consolidada de Métricas

| Métrica | Descrição | Média ($\mu$) | Desvio Padrão ($\sigma$) | Veredito |
| :--- | :--- | :--- | :--- | :--- |
| **F1-Score (LeftOf)** | Precisão das relações horizontais | **0.962** | ± 0.015 | 🟢 **Excelente** |
| **F1-Score (Below)** | Precisão das relações verticais | **0.958** | ± 0.021 | 🟢 **Excelente** |
| **Sat Agg (Global)** | Satisfação média de todos os axiomas | **0.645** | ± 0.030 | 🟡 **Correto*** |
| **Query Q1** | Existência de objeto complexo | **0.062** | ± 0.040 | ⚪ Raro |
| **Query Q2** | Existência de Cone Verde "InBetween" | **0.120** | ± 0.080 | ⚪ Raro |
| **Query Q3** | Regra: Triângulos Próximos $\to$ Mesmo Tamanho | **0.991** | ± 0.005 | 🟢 **Aprendido** |

### 2. Interpretação dos Dados (O "Porquê" dos números)

#### ✅ O Sucesso da Generalização (F1-Scores > 0.95)
Os resultados mais críticos são os **F1-Scores** para `LeftOf` e `Below`.
* O modelo treinou vendo objetos fixos (ex: um quadrado sempre na posição `0.2, 0.2`).
* No teste, ele viu objetos em posições totalmente novas e aleatórias (ex: `0.8, 0.9`).
* **Conclusão:** O fato de manter F1 > 0.95 prova que a rede neural aprendeu a **função matemática** das coordenadas ($x_1 < x_2 \implies LeftOf$) e não apenas decorou posições. A rede desacoplou a percepção visual da lógica simbólica.

#### ⚠️ O Paradoxo da Satisfação Global (Sat Agg ≈ 0.65)
Um observador desatento pode achar que 0.65 é uma nota baixa. No entanto, em Logic Tensor Networks com dados aleatórios, **isso é o comportamento correto**.
* A Base de Conhecimento inclui perguntas existenciais complexas (ex: "Existe um Cone Verde entre dois objetos?").
* Em um cenário gerado aleatoriamente, não há garantia estatística de que essa combinação específica exista.
* Quando o cenário **não** tem um Cone Verde e o modelo retorna **0.0** (Falso), ele está **acertando a lógica**. Isso reduz a média aritmética da satisfação global, mas indica precisão do agente.

#### 🧠 Raciocínio Derivado (Query Q3 ≈ 0.99)
A Query Q3 é uma regra condicional: *"Se dois triângulos estão próximos, eles têm o mesmo tamanho"*.
* O valor de **0.99** indica que o modelo aprendeu perfeitamente essa imposição.
* Isso demonstra a capacidade **Neuro-Simbólica**: o modelo ajustou os embeddings dos triângulos no espaço latente para satisfazer uma regra que conecta "Posição" (`CloseTo`) com "Atributo" (`SameSize`).

---

## 🚀 Como Executar

### Pré-requisitos
* Python 3.8+
* Bibliotecas: `torch`, `ltn`, `numpy`, `matplotlib`

### Passos
1.  Clone o repositório.
2.  Instale as dependências: `pip install ltn-torch`
3.  Execute o notebook `Trabalho_3_FIA.ipynb` em um ambiente Jupyter ou Google Colab.

---

*Desenvolvido com LTNtorch.*
