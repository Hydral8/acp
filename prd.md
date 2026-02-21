# PRD: MCP Visual ML Architecture Builder (MVP)

---

## 1. Overview

An MCP-based interactive UI utility that allows users to visually construct machine learning model architectures using draggable, connectable blocks. 

The MCP app will render an interactive visual graph editor, allow users to define network components, convert the visual graph into runnable PyTorch training code, and hand off the generated code to an external GPU provider for execution (execution layer out of scope for MVP). This is an MCP tool that renders UI dynamically via `mcp-use`, not a standalone web app.

---

## 2. Goals (MVP)

* Intuitive, beautiful block-based architecture builder
* Minimal but sufficient layer types
* Graph validation
* Code generation (PyTorch only for MVP)
* Structured output ready for remote execution

**Non-goals (for MVP):**
* Dataset hosting
* Hyperparameter sweeps
* Multi-framework support
* Versioning / collaboration

---

## 3. Target Users

* ML engineers prototyping architectures
* Researchers exploring custom networks
* Startup founders iterating model structure visually

---

## 4. Core User Flow

[Image of a drag-and-drop node-based graph editor UI]

1. User invokes MCP tool.
2. MCP renders visual graph editor.
3. User drags blocks onto canvas.
4. User connects layers.
5. User configures block parameters (modal/sidebar).
6. User selects optimizer and loss.
7. User clicks “Generate Model”.
8. MCP validates graph.
9. MCP outputs Python model class, training loop scaffold, and Config JSON.
10. Code is passed to an external GPU provider.

---

## 5. Functional Requirements

### 5.1 Visual Canvas
* Infinite scroll canvas
* Snap-to-grid
* Zoom + pan
* Dark mode default
* Smooth animations

### 5.2 Block Types (MVP)

| Category | Available Blocks |
| :--- | :--- |
| **Core Layers** | Input, Linear, Conv2D, Flatten, BatchNorm, Dropout |
| **Activations** | ReLU, GELU, Sigmoid, Tanh, Softmax |
| **Structural** | Residual Add, Concatenate |
| **Training Config** | SGD Optimizer, Adam Optimizer, MSELoss, CrossEntropyLoss |

---

## 6. Block Properties

Each block must define its core attributes. Properties are editable in the right sidebar panel.

```json
{
  "id": "string",
  "type": "string",
  "input_ports": "array",
  "output_ports": "array",
  "parameters": "object"
}

{
  "in_features": "int",
  "out_features": "int",
  "bias": "bool"
}

## 7. Graph Constraints



* Must start with exactly one Input block.
* Must end in exactly one output path.
* Must be a Directed Acyclic Graph (DAG).
* Shape compatibility validation is required (e.g., Linear input/output match, Conv2D dimension compatibility).
* Optimizer and Loss must be defined before generation.
* Errors displayed inline with visual indicators.

---

## 8. MCP Architecture

### 8.1 Tool Interface
* `render_model_builder_ui()`
* `validate_graph(graph_json)`
* `generate_pytorch_code(graph_json)`
* `export_training_package(graph_json)`

### 8.2 Interaction Pattern
User interacts visually. UI state is stored as structured graph JSON. On “Generate”, graph JSON is passed internally to `generate_pytorch_code(graph_json)`.

---

## 9. Graph JSON Schema (MVP)

```json
{
  "nodes": [
    {
      "id": "node_1",
      "type": "Linear",
      "parameters": {"in_features": 128, "out_features": 64, "bias": true}
    }
  ],
  "edges": [
    {
      "source": "node_1",
      "target": "node_2"
    }
  ],
  "optimizer": {
    "type": "Adam",
    "parameters": { "lr": 0.001 }
  },
  "loss": {
    "type": "CrossEntropyLoss"
  }
}

## 10. Code Generation (MVP)

Generated output must be clean, structured, and ready for execution.

### 10.1 Model Class
* Generates a standard PyTorch `nn.Module`.
* Layers are defined dynamically in the `__init__` method.
* Forward pass is constructed in topological order based on the visual graph.



### 10.2 Training Loop Scaffold
* Model instantiation setup.
* Optimizer and loss function instantiation.
* Dummy training step template (Dataset logic is excluded for MVP).

---

## 11. UI Design Principles

* Minimal chrome
* Strong visual hierarchy
* Soft shadows
* Rounded blocks
* Clear port indicators
* Subtle animations when connecting nodes
* Color-coded layer categories

**Layout:**
| Section | Function |
| :--- | :--- |
| **Left Panel** | Block library |
| **Center** | Interactive canvas |
| **Right Panel** | Block properties |

---

## 12. Success Criteria

* User can construct a 3–6 layer MLP visually.
* Generated PyTorch code runs without modification.
* Shape validation prevents most runtime dimension errors before generation.
* UI feels fluid and intuitive.

---

## 13. Future Extensions

* Transformer blocks
* RNN / Attention modules
* Dataset integration
* Hyperparameter tuning
* Multi-GPU / distributed config
* Save / load architectures
* Export to ONNX
* Live parameter count and FLOP estimation
* Integration with external GPU providers directly from MCP

---

## 14. Timeline (MVP)

| Phase | Deliverables |
| :--- | :--- |
| **Phase 1** | Canvas rendering, Block system, Graph JSON structure |
| **Phase 2** | Validation engine, PyTorch code generation |
| **Phase 3** | External execution interface stub (Target: functional MVP in 2–3 focused cycles) |