# Instalar todos os pacotes:
# pip install numpy pandas matplotlib scikit-learn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import threading
import time

class MLP:
    
    def __init__(self, input_size, hidden_size, output_size, activation='logística', learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Inicialização aleatória dos pesos e bias
        self.w_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b_hidden = np.zeros((1, hidden_size))
        
        self.w_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b_output = np.zeros((1, output_size))
        
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        self.activation_deriv = self._get_activation_derivative(activation)
        
        self.error_history = []
        self.training_accuracy_history = []
    
    def _get_activation(self, name):
        if name == 'logística':
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif name == 'hiperbólica':
            return lambda x: np.tanh(x)
        elif name == 'linear':
            return lambda x: x / 10
        else:
            raise ValueError(f"Função de ativação inválida: {name}")
    
    def _get_activation_derivative(self, name):
        if name == 'logística':
            def deriv(x):
                x = np.clip(x, -500, 500)
                fx = 1 / (1 + np.exp(-x))
                return fx * (1 - fx)
            return deriv
        elif name == 'hiperbólica':
            def deriv(x):
                fx = np.tanh(x)
                return 1 - fx ** 2
            return deriv
        elif name == 'linear':
            return lambda x: 1 / 10
        else:
            raise ValueError(f"Função de ativação inválida: {name}")
    
    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def feedforward(self, X):
        # Propagação para frente através da rede
        self.net_hidden = np.dot(X, self.w_input_hidden) + self.b_hidden
        self.out_hidden = self.activation(self.net_hidden)
        
        self.net_output = np.dot(self.out_hidden, self.w_hidden_output) + self.b_output
        self.out_output = self.softmax(self.net_output)
        
        return self.out_output
    
    def backpropagation(self, X, y):
        # Propagação para trás e atualização dos pesos
        self.feedforward(X)
        
        # Cálculo dos erros usando gradiente descendente
        erro_saida = self.out_output - y
        erro_oculta = np.dot(erro_saida, self.w_hidden_output.T) * self.activation_deriv(self.net_hidden)
        
        # Atualização dos pesos
        self.w_hidden_output -= self.lr * np.dot(self.out_hidden.T, erro_saida)
        self.b_output -= self.lr * np.sum(erro_saida, axis=0, keepdims=True)
        
        self.w_input_hidden -= self.lr * np.dot(X.T, erro_oculta)
        self.b_hidden -= self.lr * np.sum(erro_oculta, axis=0, keepdims=True)
        
        return np.mean(np.square(y - self.out_output))
    
    def train(self, X, y, max_epochs=3000, min_error=1e-3, verbose=True, callback=None):
        self.error_history = []
        self.training_accuracy_history = []
        
        for epoch in range(max_epochs):
            erro = self.backpropagation(X, y)
            self.error_history.append(erro)
            
            predictions = self.predict(X)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
            self.training_accuracy_history.append(accuracy)
            
            if verbose and epoch % 100 == 0:
                print(f"Época {epoch}, Erro: {erro:.6f}, Acurácia: {accuracy:.2%}")
            
            if callback:
                stop_training = callback(epoch, erro, accuracy)
                if stop_training:
                    print(f"Treinamento interrompido pelo usuário em {epoch} épocas.")
                    break
            
            if erro <= min_error:
                print(f"Treinamento encerrado por erro mínimo em {epoch} épocas.")
                break
                
        else:
            print("Treinamento encerrado por atingir o número máximo de épocas.")
        
        return epoch + 1
    
    def predict(self, X):
        return np.argmax(self.feedforward(X), axis=1)


def matriz_confusao(y_true, y_pred, classes):
    n = len(classes)
    matriz = np.zeros((n, n), dtype=int)
    
    for i in range(len(y_true)):
        matriz[y_true[i], y_pred[i]] += 1
    
    return matriz

## Classe principal da aplicação
## Cria interface
class MLPApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Rede Neural MLP - Trabalho de IA")
        self.root.geometry("1200x800")  # Aumentar altura para mostrar melhor a tabela
        
        # Configuração do estilo da interface
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        bg_color = "#f0f0f0"
        self.root.configure(bg=bg_color)
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10), borderwidth=1)
        self.style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        
        # Variáveis para dados e modelo
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.classes = None
        self.mlp = None
        self.scaler = None
        self.encoders = {}
        self.column_names = None  # Nova variável para armazenar nomes das colunas
        
        # Variáveis de controle da interface
        self.var_activation = tk.StringVar(value="logística")
        self.var_learning_rate = tk.DoubleVar(value=0.01)
        self.var_max_epochs = tk.IntVar(value=1000)
        self.var_error_limit = tk.DoubleVar(value=0.01)
        self.var_hidden_size = tk.IntVar(value=5)
        self.var_stop_criteria = tk.StringVar(value="both")
        self.var_train_test_split = tk.DoubleVar(value=0.3)
        
        self.is_training = False
        self.stop_training = False
        
        # Criação das abas da interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text='Configuração')
        
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text='Resultados')
        
        self.confusion_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.confusion_tab, text='Matriz de Confusão')
        
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text='Logs')
        
        self.setup_config_tab()
        self.setup_results_tab()
        self.setup_confusion_tab()
        self.setup_log_tab()
    
    def setup_config_tab(self):
        main_frame = ttk.Frame(self.config_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Seção de carregamento de arquivos
        file_frame = ttk.LabelFrame(main_frame, text="Carregar Dados")
        file_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(file_frame, text="Carregar Arquivo Único", command=self.load_single_file).pack(side='left', padx=10, pady=10)
        ttk.Button(file_frame, text="Carregar Treino e Teste", command=self.load_train_test_files).pack(side='left', padx=10, pady=10)
        
        split_frame = ttk.Frame(file_frame)
        split_frame.pack(side='right', padx=10, pady=10)
        
        ttk.Label(split_frame, text="% para Teste:").pack(side='left')
        split_scale = ttk.Scale(split_frame, from_=0.1, to=0.5, variable=self.var_train_test_split, orient='horizontal', length=100)
        split_scale.pack(side='left', padx=5)
        ttk.Label(split_frame, textvariable=self.var_train_test_split).pack(side='left')
        
        self.file_info = ttk.Label(file_frame, text="Nenhum arquivo carregado")
        self.file_info.pack(side='bottom', fill='x', padx=10, pady=5)
        
        # Configurações da rede neural
        network_frame = ttk.LabelFrame(main_frame, text="Configurações da Rede")
        network_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(network_frame, text="Função de Ativação:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        activation_combo = ttk.Combobox(network_frame, textvariable=self.var_activation, values=["logística", "hiperbólica", "linear"], state="readonly", width=15)
        activation_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(network_frame, text="Taxa de Aprendizagem:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        lr_frame = ttk.Frame(network_frame)
        lr_frame.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        lr_scale = ttk.Scale(lr_frame, from_=0.001, to=1.0, variable=self.var_learning_rate, orient='horizontal', length=150)
        lr_scale.pack(side='left')
        lr_entry = ttk.Entry(lr_frame, textvariable=self.var_learning_rate, width=6)
        lr_entry.pack(side='left', padx=5)
        
        # Configuração da camada oculta
        hidden_frame = ttk.LabelFrame(network_frame, text="Neurônios na Camada Oculta")
        hidden_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='we')
        
        self.hidden_option = tk.StringVar(value="arithmetic")
        
        ttk.Radiobutton(hidden_frame, text="Média Aritmética", variable=self.hidden_option, value="arithmetic").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Radiobutton(hidden_frame, text="Média Geométrica", variable=self.hidden_option, value="geometric").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        ttk.Radiobutton(hidden_frame, text="Personalizado", variable=self.hidden_option, value="custom").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        
        self.custom_neurons = tk.IntVar(value=5)
        ttk.Entry(hidden_frame, textvariable=self.custom_neurons, width=5).grid(row=2, column=1, padx=5, pady=2, sticky='w')
        
        self.layers_info = ttk.Label(hidden_frame, text="")
        self.layers_info.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        
        # Critérios de parada
        stop_frame = ttk.LabelFrame(main_frame, text="Critérios de Parada")
        stop_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(stop_frame, text="Por Iteração", variable=self.var_stop_criteria, value="epochs").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Radiobutton(stop_frame, text="Por Erro", variable=self.var_stop_criteria, value="error").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        ttk.Radiobutton(stop_frame, text="Ambos", variable=self.var_stop_criteria, value="both").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        
        ttk.Label(stop_frame, text="Máximo de Épocas:").grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(stop_frame, textvariable=self.var_max_epochs, width=8).grid(row=0, column=2, padx=5, pady=5, sticky='w')
        
        ttk.Label(stop_frame, text="Limiar de Erro:").grid(row=1, column=1, padx=5, pady=5, sticky='w')
        ttk.Entry(stop_frame, textvariable=self.var_error_limit, width=8).grid(row=1, column=2, padx=5, pady=5, sticky='w')
        
        # Botões de controle do treinamento
        train_frame = ttk.Frame(main_frame)
        train_frame.pack(fill='x', padx=5, pady=10)
        
        self.train_button = ttk.Button(train_frame, text="Treinar Rede", command=self.train_network, style='Accent.TButton')
        self.train_button.pack(side='left', padx=10)
        
        self.style.configure('Accent.TButton', foreground='white', background='#0078d7')
        self.style.map('Accent.TButton', 
                     foreground=[('pressed', 'white'), ('active', 'white')],
                     background=[('pressed', '#005a9e'), ('active', '#1a86d6')])
        
        self.stop_button = ttk.Button(train_frame, text="Parar Treinamento", command=self.stop_training_process, state='disabled')
        self.stop_button.pack(side='left', padx=10)
        
        self.progress = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=5)
        
        # Quadro de dados em tempo real
        progress_frame = ttk.LabelFrame(main_frame, text="Dados do Treinamento")
        progress_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Status atual
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(status_frame, text="Época:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.current_epoch = ttk.Label(status_frame, text="0", font=('Arial', 10, 'bold'), foreground='blue')
        self.current_epoch.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        
        ttk.Label(status_frame, text="Erro:").grid(row=0, column=2, padx=15, pady=2, sticky='w')
        self.current_error = ttk.Label(status_frame, text="0.000000", font=('Arial', 10, 'bold'), foreground='red')
        self.current_error.grid(row=0, column=3, padx=5, pady=2, sticky='w')
        
        ttk.Label(status_frame, text="Acurácia:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.current_accuracy = ttk.Label(status_frame, text="0.00%", font=('Arial', 10, 'bold'), foreground='green')
        self.current_accuracy.grid(row=1, column=1, padx=5, pady=2, sticky='w')
        
        ttk.Label(status_frame, text="Status:").grid(row=1, column=2, padx=15, pady=2, sticky='w')
        self.training_status = ttk.Label(status_frame, text="Aguardando", font=('Arial', 10, 'bold'))
        self.training_status.grid(row=1, column=3, padx=5, pady=2, sticky='w')
        
        # Tabela de dados de exemplo - será criada dinamicamente
        self.table_frame = ttk.Frame(progress_frame)
        self.table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_tree = None  # Será criado quando os dados forem carregados
        
        # Botões
        btn_frame = ttk.Frame(progress_frame)
        btn_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(btn_frame, text="Mostrar Dados de Treino", command=self.show_training_data).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Mostrar Dados de Teste", command=self.show_test_data).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Limpar Tabela", command=self.clear_data_table).pack(side='left', padx=5)
    
    def create_data_table(self, columns):
        """Cria a tabela dinamicamente baseada nas colunas do dataset."""
        # Remover tabela anterior se existir
        if self.data_tree:
            self.data_tree.destroy()
        
        # Limpar frame
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Criar nova tabela com as colunas corretas
        self.data_tree = ttk.Treeview(self.table_frame, columns=columns, show='headings', height=10)
        
        # Configurar cabeçalhos e larguras das colunas
        for col in columns:
            self.data_tree.heading(col, text=str(col))
            # Ajustar largura baseada no tipo de coluna
            if col == columns[-1]:  # Última coluna (assumindo que é a classe)
                self.data_tree.column(col, width=100, anchor='center')
            else:
                self.data_tree.column(col, width=80, anchor='center')
        
        # Scrollbar para a tabela
        data_scrollbar = ttk.Scrollbar(self.table_frame, orient='vertical', command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=data_scrollbar.set)
        
        self.data_tree.pack(side='left', fill='both', expand=True)
        data_scrollbar.pack(side='right', fill='y')
        
        self.log(f"Tabela criada com {len(columns)} colunas: {columns}")
    
    def setup_results_tab(self):
        main_frame = ttk.Frame(self.results_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Resumo do treinamento
        summary_frame = ttk.LabelFrame(main_frame, text="Resumo do Treinamento")
        summary_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(summary_frame, text="Épocas:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.epochs_value = ttk.Label(summary_frame, text="N/A")
        self.epochs_value.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(summary_frame, text="Erro Final:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.error_value = ttk.Label(summary_frame, text="N/A")
        self.error_value.grid(row=1, column=1, padx=10, pady=5, sticky='w')
        
        ttk.Label(summary_frame, text="Acurácia de Treino:").grid(row=0, column=2, padx=10, pady=5, sticky='w')
        self.train_acc_value = ttk.Label(summary_frame, text="N/A")
        self.train_acc_value.grid(row=0, column=3, padx=10, pady=5, sticky='w')
        
        ttk.Label(summary_frame, text="Acurácia de Teste:").grid(row=1, column=2, padx=10, pady=5, sticky='w')
        self.test_acc_value = ttk.Label(summary_frame, text="N/A")
        self.test_acc_value.grid(row=1, column=3, padx=10, pady=5, sticky='w')
        
        # Frames para gráficos
        charts_frame = ttk.Frame(main_frame)
        charts_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.error_chart_frame = ttk.LabelFrame(charts_frame, text="Evolução do Erro")
        self.error_chart_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        self.accuracy_chart_frame = ttk.LabelFrame(charts_frame, text="Evolução da Acurácia")
        self.accuracy_chart_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
    
    def setup_confusion_tab(self):
        main_frame = ttk.Frame(self.confusion_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        title_label = ttk.Label(main_frame, text="Matriz de Confusão", font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        self.confusion_matrix_frame = ttk.Frame(main_frame)
        self.confusion_matrix_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.metrics_frame = ttk.LabelFrame(main_frame, text="Métricas por Classe")
        self.metrics_frame.pack(fill='x', padx=10, pady=10)
    
    def setup_log_tab(self):
        main_frame = ttk.Frame(self.log_tab)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, wrap='word', height=20)
        self.log_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        ttk.Button(main_frame, text="Limpar Logs", command=self.clear_logs).pack(pady=5)
    
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert('end', log_entry)
        self.log_text.see('end')
        print(message)
    
    def clear_logs(self):
        self.log_text.delete('1.0', 'end')
        self.log("Logs limpos.")
    
    def clear_data_table(self):
        """Limpa a tabela de dados."""
        if self.data_tree:
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            self.log("Tabela de dados limpa.")
        else:
            self.log("Nenhuma tabela para limpar.")
    
    def show_training_data(self):
        """Mostra alguns dados de treinamento na tabela."""
        if self.X_train is None:
            messagebox.showwarning("Aviso", "Carregue os dados primeiro.")
            return
        
        if not self.data_tree:
            messagebox.showwarning("Aviso", "Tabela não foi criada. Carregue os dados primeiro.")
            return
        
        self.log("Mostrando dados de treino na tabela...")
        self.clear_data_table()
        
        try:
            # Tentar usar dados originais se disponíveis
            if hasattr(self, 'original_train_data'):
                df = self.original_train_data
                self.log(f"Usando dados originais de treino ({len(df)} linhas)")
                for i in range(min(20, len(df))):
                    row = df.iloc[i]
                    # Converter valores para formato apropriado
                    values = []
                    for j, val in enumerate(row):
                        if j < len(row) - 1:  # Colunas de features
                            try:
                                values.append(f"{float(val):.2f}")
                            except:
                                values.append(str(val))
                        else:  # Coluna de classe
                            values.append(str(val))
                    
                    self.data_tree.insert('', 'end', values=tuple(values))
            else:
                # Usar dados processados (desnormalizados)
                self.log("Usando dados processados de treino")
                if self.scaler is not None:
                    for i in range(min(20, len(self.X_train))):
                        original_data = self.scaler.inverse_transform([self.X_train[i]])[0]
                        classe_idx = np.argmax(self.y_train[i])
                        classe = self.classes[classe_idx]
                        
                        values = [f"{val:.2f}" for val in original_data] + [classe]
                        self.data_tree.insert('', 'end', values=tuple(values))
            
            # Contar quantas linhas foram inseridas
            children = self.data_tree.get_children()
            self.log(f"Tabela populada com {len(children)} linhas")
                        
        except Exception as e:
            self.log(f"Erro ao mostrar dados de treino: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao mostrar dados de treino: {str(e)}")
    
    def show_test_data(self):
        """Mostra alguns dados de teste na tabela."""
        if self.X_test is None:
            messagebox.showwarning("Aviso", "Carregue os dados primeiro.")
            return
        
        if not self.data_tree:
            messagebox.showwarning("Aviso", "Tabela não foi criada. Carregue os dados primeiro.")
            return
        
        self.log("Mostrando dados de teste na tabela...")
        self.clear_data_table()
        
        try:
            # Tentar usar dados originais se disponíveis
            if hasattr(self, 'original_test_data'):
                df = self.original_test_data
                self.log(f"Usando dados originais de teste ({len(df)} linhas)")
                for i in range(min(20, len(df))):
                    row = df.iloc[i]
                    # Converter valores para formato apropriado
                    values = []
                    for j, val in enumerate(row):
                        if j < len(row) - 1:  # Colunas de features
                            try:
                                values.append(f"{float(val):.2f}")
                            except:
                                values.append(str(val))
                        else:  # Coluna de classe
                            values.append(str(val))
                    
                    self.data_tree.insert('', 'end', values=tuple(values))
            else:
                # Usar dados processados
                self.log("Usando dados processados de teste")
                if self.scaler is not None:
                    for i in range(min(20, len(self.X_test))):
                        original_data = self.scaler.inverse_transform([self.X_test[i]])[0]
                        classe_idx = np.argmax(self.y_test[i])
                        classe = self.classes[classe_idx]
                        
                        values = [f"{val:.2f}" for val in original_data] + [classe]
                        self.data_tree.insert('', 'end', values=tuple(values))
            
            # Contar quantas linhas foram inseridas
            children = self.data_tree.get_children()
            self.log(f"Tabela populada com {len(children)} linhas")
                        
        except Exception as e:
            self.log(f"Erro ao mostrar dados de teste: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao mostrar dados de teste: {str(e)}")
    
    def update_training_display(self, epoch, error, accuracy):
        """Atualiza apenas o display de status durante o treinamento."""
        self.current_epoch.config(text=str(epoch))
        self.current_error.config(text=f"{error:.6f}")
        self.current_accuracy.config(text=f"{accuracy:.2%}")
    
    def load_single_file(self):
        # Carrega um arquivo CSV e divide automaticamente em treino e teste
        filename = filedialog.askopenfilename(title="Selecione o arquivo CSV", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if not filename:
            return
        
        try:
            self.log(f"Carregando arquivo: {filename}")
            
            df = pd.read_csv(filename)
            self.log(f"Arquivo carregado com {len(df)} linhas e {len(df.columns)} colunas")
            self.log(f"Colunas: {list(df.columns)}")
            
            # Armazenar nomes das colunas
            self.column_names = list(df.columns)
            
            # Criar tabela dinamicamente
            self.create_data_table(self.column_names)
            
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            test_size = self.var_train_test_split.get()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Criar DataFrames originais divididos para exibição
            train_indices = X_train.index
            test_indices = X_test.index
            self.original_train_data = df.loc[train_indices].reset_index(drop=True)
            self.original_test_data = df.loc[test_indices].reset_index(drop=True)
            
            X_train_proc, X_test_proc, y_train_enc, y_test_enc, classes = self.process_data(X_train, X_test, y_train, y_test)
            
            self.X_train = X_train_proc
            self.X_test = X_test_proc
            self.y_train = y_train_enc
            self.y_test = y_test_enc
            self.classes = classes
            
            num_features = X.shape[1]
            num_classes = len(classes)
            num_train = len(X_train)
            num_test = len(X_test)
            
            info_text = f"Arquivo: {os.path.basename(filename)}\n" + \
                       f"Exemplos: {num_train} (treino), {num_test} (teste)\n" + \
                       f"Atributos: {num_features}, Classes: {num_classes}"
            
            self.file_info.config(text=info_text)
            
            hidden_info = self.calculate_hidden_neurons(num_features, num_classes)
            
            self.layers_info.config(
                text=f"Entradas: {num_features}, Saídas: {num_classes}\n" + 
                     f"Média Aritmética: {hidden_info['arithmetic']}, " +
                     f"Média Geométrica: {hidden_info['geometric']}"
            )
            
            self.custom_neurons.set(hidden_info['arithmetic'])
            
            self.log(f"Dados carregados: {num_train} exemplos para treino, {num_test} para teste, {num_classes} classes")
            
            # Mostrar dados na tabela automaticamente
            self.show_training_data()
            
        except Exception as e:
            self.log(f"Erro ao carregar o arquivo: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao carregar o arquivo:\n{str(e)}")
    
    def load_train_test_files(self):
        # Carrega arquivos separados para treino e teste
        train_file = filedialog.askopenfilename(title="Selecione o arquivo de TREINO", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if not train_file:
            return
        
        test_file = filedialog.askopenfilename(title="Selecione o arquivo de TESTE", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if not test_file:
            return
        
        try:
            self.log(f"Carregando arquivos: {train_file} e {test_file}")
            
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
            
            self.log(f"Arquivo de treino: {len(df_train)} linhas, {len(df_train.columns)} colunas")
            self.log(f"Arquivo de teste: {len(df_test)} linhas, {len(df_test.columns)} colunas")
            
            if not df_train.columns.equals(df_test.columns):
                raise ValueError("As colunas dos arquivos de treino e teste não coincidem")
            
            # Armazenar nomes das colunas
            self.column_names = list(df_train.columns)
            
            # Criar tabela dinamicamente
            self.create_data_table(self.column_names)
            
            # Salvar dados originais para exibição
            self.original_train_data = df_train.copy()
            self.original_test_data = df_test.copy()
            
            X_train = df_train.iloc[:, :-1]
            y_train = df_train.iloc[:, -1]
            
            X_test = df_test.iloc[:, :-1]
            y_test = df_test.iloc[:, -1]
            
            X_train_proc, X_test_proc, y_train_enc, y_test_enc, classes = self.process_data(X_train, X_test, y_train, y_test)
            
            self.X_train = X_train_proc
            self.X_test = X_test_proc
            self.y_train = y_train_enc
            self.y_test = y_test_enc
            self.classes = classes
            
            num_features = X_train.shape[1]
            num_classes = len(classes)
            num_train = len(X_train)
            num_test = len(X_test)
            
            info_text = f"Arquivos: {os.path.basename(train_file)}, {os.path.basename(test_file)}\n" + \
                       f"Exemplos: {num_train} (treino), {num_test} (teste)\n" + \
                       f"Atributos: {num_features}, Classes: {num_classes}"
            
            self.file_info.config(text=info_text)
            
            hidden_info = self.calculate_hidden_neurons(num_features, num_classes)
            
            self.layers_info.config(
                text=f"Entradas: {num_features}, Saídas: {num_classes}\n" + 
                     f"Média Aritmética: {hidden_info['arithmetic']}, " +
                     f"Média Geométrica: {hidden_info['geometric']}"
            )
            
            self.custom_neurons.set(hidden_info['arithmetic'])
            
            self.log(f"Dados carregados: {num_train} exemplos para treino, {num_test} para teste, {num_classes} classes")
            
            # Mostrar automaticamente dados de treino
            self.show_training_data()
            
        except Exception as e:
            self.log(f"Erro ao carregar os arquivos: {str(e)}")
            messagebox.showerror("Erro", f"Erro ao carregar os arquivos:\n{str(e)}")
    
    def process_data(self, X_train, X_test, y_train, y_test):
        # Processamento dos dados: encoding categórico e normalização
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        self.encoders = {}
        colunas_cat = X_train.select_dtypes(include='object').columns.tolist()
        
        # Codificação de variáveis categóricas
        for col in colunas_cat:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            self.encoders[col] = le
        
        # Normalização dos dados
        self.scaler = StandardScaler()
        X_train_proc = self.scaler.fit_transform(X_train)
        X_test_proc = self.scaler.transform(X_test)
        
        # One-hot encoding para as classes de saída
        y_train_enc = pd.get_dummies(y_train)
        classes = y_train_enc.columns.tolist()
        
        y_test_enc = pd.get_dummies(y_test)
        y_test_enc = y_test_enc.reindex(columns=classes, fill_value=0)
        
        return X_train_proc, X_test_proc, y_train_enc.values, y_test_enc.values, classes
    
    def calculate_hidden_neurons(self, input_size, output_size):
        arithmetic = math.ceil((input_size + output_size) / 2)
        geometric = math.ceil(math.sqrt(input_size * output_size))
        
        return {
            "arithmetic": arithmetic,
            "geometric": geometric
        }
    
    def train_network(self):
        # Configura e inicia o treinamento da rede neural
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Aviso", "Por favor, carregue os dados primeiro.")
            return
        
        activation = self.var_activation.get()
        learning_rate = self.var_learning_rate.get()
        
        # Determinar número de neurônios na camada oculta
        hidden_option = self.hidden_option.get()
        if hidden_option == "arithmetic":
            hidden_size = math.ceil((self.X_train.shape[1] + self.y_train.shape[1]) / 2)
        elif hidden_option == "geometric":
            hidden_size = math.ceil(math.sqrt(self.X_train.shape[1] * self.y_train.shape[1]))
        else:
            hidden_size = self.custom_neurons.get()
            if hidden_size <= 0:
                messagebox.showwarning("Aviso", "Número de neurônios deve ser maior que zero.")
                return
        
        # Configurar critérios de parada
        stop_criteria = self.var_stop_criteria.get()
        max_epochs = self.var_max_epochs.get()
        error_limit = self.var_error_limit.get()
        
        if stop_criteria == "epochs":
            actual_error_limit = 0
            actual_max_epochs = max_epochs
        elif stop_criteria == "error":
            actual_error_limit = error_limit
            actual_max_epochs = 100000
        else:
            actual_error_limit = error_limit
            actual_max_epochs = max_epochs
        
        self.log(f"Criando rede MLP com {hidden_size} neurônios na camada oculta")
        self.log(f"Configurações: ativação={activation}, taxa={learning_rate:.4f}, " + 
               f"épocas_max={actual_max_epochs}, erro_min={actual_error_limit:.6f}")
        
        input_size = self.X_train.shape[1]
        output_size = self.y_train.shape[1]
        
        self.mlp = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            learning_rate=learning_rate
        )
        
        # Configurar interface para treinamento
        self.is_training = True
        self.stop_training = False
        
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress["value"] = 0
        
        # Preparar display de treinamento
        self.current_epoch.config(text="0")
        self.current_error.config(text="0.000000")
        self.current_accuracy.config(text="0.00%")
        self.training_status.config(text="Iniciando...")
        
        # Manter na aba de configuração para ver o progresso
        self.notebook.select(0)
        
        # Iniciar thread de treinamento
        train_thread = threading.Thread(target=self._train_thread, args=(actual_max_epochs, actual_error_limit))
        train_thread.daemon = True
        train_thread.start()
    
    def _train_thread(self, max_epochs, min_error):
        # Executa o treinamento em thread separada para não travar a interface
        try:
            start_time = time.time()
            
            def update_callback(epoch, error, accuracy):
                # Atualizar barra de progresso
                progress = min(100, int(epoch / max_epochs * 100))
                self.progress["value"] = progress
                
                # Atualizar status de treinamento
                self.root.after(0, lambda: self.training_status.config(text="Treinando..."))
                
                # Atualizar display de progresso em tempo real
                self.root.after(0, self.update_training_display, epoch, error, accuracy)
                
                # Log menos frequente para não sobrecarregar
                if epoch % 100 == 0:
                    self.log(f"Época {epoch}, Erro: {error:.6f}, Acurácia: {accuracy:.2%}")
                
                return self.stop_training
            
            epochs_completed = self.mlp.train(self.X_train, self.y_train, max_epochs=max_epochs, min_error=min_error, verbose=False, callback=update_callback)
            training_time = time.time() - start_time
            
            # Calcular métricas finais
            train_preds = self.mlp.predict(self.X_train)
            train_true = np.argmax(self.y_train, axis=1)
            train_accuracy = np.mean(train_preds == train_true)
            
            test_preds = self.mlp.predict(self.X_test)
            test_true = np.argmax(self.y_test, axis=1)
            test_accuracy = np.mean(test_preds == test_true)
            
            final_error = self.mlp.error_history[-1] if self.mlp.error_history else float('inf')
            
            # Atualizar status final
            self.root.after(0, lambda: self.training_status.config(text="Concluído ✓"))
            
            self.log(f"Treinamento concluído em {epochs_completed} épocas ({training_time:.2f} segundos)")
            self.log(f"Erro final: {final_error:.6f}")
            self.log(f"Acurácia de treino: {train_accuracy:.2%}")
            self.log(f"Acurácia de teste: {test_accuracy:.2%}")
            
            self.root.after(100, self._update_ui_after_training, {
                "epochs": epochs_completed,
                "error": final_error,
                "train_acc": train_accuracy,
                "test_acc": test_accuracy
            })
            
        except Exception as e:
            self.log(f"Erro durante o treinamento: {str(e)}")
            self.root.after(0, lambda: self.training_status.config(text="Erro ✗"))
            self.root.after(100, self._restore_ui_after_error)
        
        finally:
            self.is_training = False
    
    def _update_ui_after_training(self, results):
        # Atualiza a interface com os resultados do treinamento
        self.epochs_value.config(text=str(results["epochs"]))
        self.error_value.config(text=f"{results['error']:.6f}")
        self.train_acc_value.config(text=f"{results['train_acc']:.2%}")
        self.test_acc_value.config(text=f"{results['test_acc']:.2%}")
        
        self.plot_training_charts()
        self.plot_confusion_matrix()
        
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress["value"] = 100
        
        # Ir para aba de resultados após completar
        self.notebook.select(1)
    
    def _restore_ui_after_error(self):
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.training_status.config(text="Erro ✗")
        self.current_epoch.config(text="0")
        self.current_error.config(text="0.000000")
        self.current_accuracy.config(text="0.00%")
    
    def stop_training_process(self):
        if self.is_training:
            self.stop_training = True
            self.training_status.config(text="Parando...")
            self.log("Interrompendo treinamento...")
    
    def plot_training_charts(self):
        # Plota gráficos de evolução do erro e acurácia
        if self.mlp is None or not self.mlp.error_history:
            return
        
        for widget in self.error_chart_frame.winfo_children():
            widget.destroy()
        
        for widget in self.accuracy_chart_frame.winfo_children():
            widget.destroy()
        
        # Gráfico de erro
        fig_error = plt.Figure(figsize=(5, 4), dpi=100)
        ax_error = fig_error.add_subplot(111)
        
        ax_error.plot(self.mlp.error_history, 'r-')
        ax_error.set_title('Evolução do Erro')
        ax_error.set_xlabel('Época')
        ax_error.set_ylabel('Erro Quadrático Médio')
        ax_error.grid(True)
        
        canvas_error = FigureCanvasTkAgg(fig_error, self.error_chart_frame)
        canvas_error.draw()
        canvas_error.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico de acurácia
        if self.mlp.training_accuracy_history:
            fig_acc = plt.Figure(figsize=(5, 4), dpi=100)
            ax_acc = fig_acc.add_subplot(111)
            
            ax_acc.plot(self.mlp.training_accuracy_history, 'g-')
            ax_acc.set_title('Evolução da Acurácia')
            ax_acc.set_xlabel('Época')
            ax_acc.set_ylabel('Acurácia')
            ax_acc.grid(True)
            ax_acc.set_ylim(0, 1)
            
            canvas_acc = FigureCanvasTkAgg(fig_acc, self.accuracy_chart_frame)
            canvas_acc.draw()
            canvas_acc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_confusion_matrix(self):
        # Plota a matriz de confusão e calcula métricas
        if self.mlp is None or self.X_test is None:
            return
        
        for widget in self.confusion_matrix_frame.winfo_children():
            widget.destroy()
        
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        y_pred = self.mlp.predict(self.X_test)
        y_true = np.argmax(self.y_test, axis=1)
        
        conf_matrix = matriz_confusao(y_true, y_pred, self.classes)
        
        fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)
        
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        ax.set_xticks(np.arange(len(self.classes)))
        ax.set_yticks(np.arange(len(self.classes)))
        ax.set_xticklabels(self.classes, rotation=45, ha="right")
        ax.set_yticklabels(self.classes)
        
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                ax.text(j, i, str(conf_matrix[i, j]),
                       ha="center", va="center",
                       color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
        
        ax.set_title('Matriz de Confusão')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.confusion_matrix_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.calculate_metrics(conf_matrix)
    
    def calculate_metrics(self, conf_matrix):
        # Calcula métricas de classificação por classe
        headers = ["Classe", "Precisão", "Recall", "F1", "Acurácia"]
        
        for i, header in enumerate(headers):
            ttk.Label(self.metrics_frame, text=header, font=('Arial', 10, 'bold')).grid(row=0, column=i, padx=10, pady=5)
        
        for i, classe in enumerate(self.classes):
            tp = conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp
            
            precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0
            acuracia = tp / np.sum(conf_matrix[i, :]) if np.sum(conf_matrix[i, :]) > 0 else 0
            
            ttk.Label(self.metrics_frame, text=classe).grid(row=i+1, column=0, padx=10, pady=2)
            ttk.Label(self.metrics_frame, text=f"{precisao:.2%}").grid(row=i+1, column=1, padx=10, pady=2)
            ttk.Label(self.metrics_frame, text=f"{recall:.2%}").grid(row=i+1, column=2, padx=10, pady=2)
            ttk.Label(self.metrics_frame, text=f"{f1:.2%}").grid(row=i+1, column=3, padx=10, pady=2)
            ttk.Label(self.metrics_frame, text=f"{acuracia:.2%}").grid(row=i+1, column=4, padx=10, pady=2)

def main():
    root = tk.Tk()
    app = MLPApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()