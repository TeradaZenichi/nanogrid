# Manual rápido: tmux + Gurobi no JupyterLab

Assumindo que você já está dentro do JupyterLab do SageMaker.

## 1. Abrir um terminal

No JupyterLab: menu **File → New → Terminal** (ou o ícone **+** → *Terminal*).

## 2. Entrar no projeto

```bash
cd /home/ec2-user/SageMaker/nanogrid
```

## 3. Criar a sessão tmux

```bash
tmux new -s campanha
```

A tela muda e aparece uma barra verde embaixo: você está dentro do tmux.
Tudo que rodar aqui continua vivo mesmo se fechar o navegador.

## 4. Conferir o Gurobi (dentro do tmux)

```bash
export GRB_LICENSE_FILE=/home/ec2-user/SageMaker/gurobi.lic
python -c "from opt.utils import detect_solver; print(detect_solver())"
```

- Imprimiu **`gurobi`** → licença OK, pode rodar.
- Imprimiu **`appsi_highs`** com aviso "size-limited" → a licença não foi
  encontrada. Confira o caminho do arquivo:
  ```bash
  ls -l /home/ec2-user/SageMaker/gurobi.lic
  ```
  Se não existir, ache com `find /home/ec2-user -name gurobi.lic 2>/dev/null`
  e ajuste o `export`.

## 5. Rodar o experimento

```bash
python 1-sizing.py
```

Ou encadear vários em sequência:

```bash
python 1-sizing.py && python 2-forecast_eval.py
```

## 6. Soltar a sessão (deixar rodando) e fechar o navegador

Pressione, em sequência:

```
Ctrl+B   (solta as teclas)   depois   D
```

Aparece `[detached]`. Agora pode fechar o navegador — o processo continua.

## 7. Voltar depois

Abra um terminal novo no JupyterLab e:

```bash
tmux attach -t campanha     # reconecta e mostra a saída ao vivo
tmux ls                     # (se esqueceu o nome) lista as sessões
```

## Atalhos (sempre Ctrl+B, solta, depois a tecla)

| Ação | Teclas |
|---|---|
| Soltar a sessão (manter rodando) | `Ctrl+B` então `D` |
| Rolar a tela para ver log antigo | `Ctrl+B` então `[` (sair: `q`) |
| Reconectar | `tmux attach -t campanha` |
| Encerrar a sessão | dentro dela: `exit` |

## Dois lembretes

1. **Licença**: a cada sessão tmux nova, refaça o `export` do passo 4 — ou
   grave uma vez para sempre:
   ```bash
   echo 'export GRB_LICENSE_FILE=/home/ec2-user/SageMaker/gurobi.lic' >> ~/.bashrc
   ```
2. **tmux protege contra desconexão, não contra a instância ser parada.**
   Se houver auto-shutdown por ociosidade, desative-o antes de runs longas
   (Notebook instance → Lifecycle configuration). Mesmo assim, o
   `2-forecast_eval.py` retoma de onde parou pelo checkpoint.
