export const planData = {
  phases: [
    {
      id: "phase-1",
      title: "Phase 1: Python & Math Foundations",
      subtitle: "Jan 2026 - Mar 2026 (1hr/day = 7hr/week)",
      months: [
        {
          id: "jan-2026",
          name: "January 2026",
          weeks: [
            {
              id: "jan-26-w1",
              title: "Week 1: Jan 18-24",
              focus: "Python OOP - Classes & Objects",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you create a class with __init__, use self correctly, and explain instance vs class variables?",
              reference: "realpython.com/python3-object-oriented-programming/",
              tasks: [
                "Read: Real Python OOP Tutorial (Classes section)",
                "Code: Create a Book class with title, author, pages",
                "Code: Create a Library class that holds multiple Books",
                "LeetCode: Two Sum (#1)",
                "LeetCode: Valid Parentheses (#20)",
                "Depth Check: Explain what 'self' does and why it's needed"
              ]
            },
            {
              id: "jan-26-w2",
              title: "Week 2: Jan 25-31",
              focus: "Python OOP - Inheritance & Methods",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Can you use super() and explain method resolution order?",
              reference: "docs.python.org/3/tutorial/classes.html",
              tasks: [
                "Read: Python Docs - Inheritance section",
                "Code: Create subclasses (EBook extends Book, Member extends User)",
                "Code: Implement __str__ and __repr__ magic methods",
                "LeetCode: Palindrome Number (#9)",
                "LeetCode: Contains Duplicate (#217)",
                "LeetCode: Valid Anagram (#242)",
                "Project: Complete Library Management System"
              ]
            }
          ]
        },
        {
          id: "feb-2026",
          name: "February 2026",
          weeks: [
            {
              id: "feb-26-w1",
              title: "Week 1: Feb 1-7",
              focus: "Python Advanced - Comprehensions & Decorators",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Can you write a nested list comprehension and explain @decorator syntax?",
              reference: "Fluent Python Ch 7-9 (O'Reilly)",
              tasks: [
                "Read: List/Dict/Set comprehensions guide",
                "Code: Rewrite 5 for-loops as comprehensions",
                "Read: Decorators tutorial (basics only)",
                "Code: Write a simple timing decorator",
                "LeetCode: Best Time to Buy/Sell Stock (#121)",
                "LeetCode: Merge Two Sorted Lists (#21)",
                "LeetCode: Climbing Stairs (#70)"
              ]
            },
            {
              id: "feb-26-w2",
              title: "Week 2: Feb 8-14",
              focus: "NumPy Fundamentals",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you explain broadcasting rules and why vectorization is faster than loops?",
              reference: "numpy.org/doc/stable/user/quickstart.html",
              tasks: [
                "Read: NumPy Quickstart (creation, indexing)",
                "Code: Create arrays with zeros, ones, arange, linspace",
                "Read: Broadcasting rules",
                "Code: Normalize a (1000, 784) image array without loops",
                "LeetCode: Remove Duplicates Sorted Array (#26)",
                "LeetCode: Search Insert Position (#35)"
              ]
            },
            {
              id: "feb-26-w3",
              title: "Week 3: Feb 15-21",
              focus: "Linear Algebra Foundations",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Can you multiply a (3,4) × (4,2) matrix by hand and know the output shape?",
              reference: "3blue1brown.com/topics/linear-algebra",
              tasks: [
                "Watch: 3Blue1Brown - Vectors (Ep 1-3)",
                "Code: Implement dot product in NumPy",
                "Watch: 3Blue1Brown - Matrix Multiplication (Ep 4-5)",
                "Code: Implement matrix multiply WITHOUT np.dot",
                "LeetCode: Maximum Subarray (#53)",
                "LeetCode: Plus One (#66)"
              ]
            },
            {
              id: "feb-26-w4",
              title: "Week 4: Feb 22-28",
              focus: "Linear Algebra + Calculus Intro",
              hours: 7,
              depth: "🟢 Awareness",
              depthCheck: "Can you explain what eigenvalues represent geometrically?",
              reference: "mml-book.github.io (Ch 2-3)",
              tasks: [
                "Watch: 3Blue1Brown - Eigenvalues (Ep 14)",
                "Read: MML Book Ch 2 (skim for intuition)",
                "Watch: 3Blue1Brown - Derivatives Intro",
                "Code: Compute gradient of f(x,y) = x²y by hand",
                "LeetCode: Sqrt(x) (#69)",
                "LeetCode: Merge Sorted Array (#88)",
                "GitHub: Push first project (Library System)"
              ]
            }
          ]
        },
        {
          id: "mar-2026",
          name: "March 2026",
          weeks: [
            {
              id: "mar-26-w1",
              title: "Week 1: Mar 1-7",
              focus: "Calculus for ML",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Can you apply chain rule and compute partial derivatives?",
              reference: "deeplearningbook.org Ch 4",
              tasks: [
                "Watch: 3Blue1Brown - Chain Rule",
                "Read: Deep Learning Book Ch 4.3 (Gradient-Based Optimization)",
                "Practice: Compute ∂f/∂x for 5 different functions",
                "LeetCode: Single Number (#136)",
                "LeetCode: Linked List Cycle (#141)",
                "LeetCode: Min Stack (#155)"
              ]
            },
            {
              id: "mar-26-w2",
              title: "Week 2: Mar 8-14",
              focus: "Probability & Statistics",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Can you explain Bayes theorem with a real example?",
              reference: "StatQuest YouTube (Probability playlist)",
              tasks: [
                "Watch: StatQuest - Probability Basics (3 videos)",
                "Watch: StatQuest - Bayes Theorem",
                "Practice: Solve 5 conditional probability problems",
                "LeetCode: Intersection of Two Linked Lists (#160)",
                "LeetCode: Two Sum II (#167)",
                "LeetCode: Majority Element (#169)"
              ]
            },
            {
              id: "mar-26-w3",
              title: "Week 3: Mar 15-21",
              focus: "Google Data Analytics Cert (Week 1-3)",
              hours: 7,
              depth: "Certificate",
              depthCheck: "N/A - Complete course modules",
              reference: "coursera.org/professional-certificates/google-data-analytics",
              tasks: [
                "Enroll: Google Data Analytics Certificate",
                "Complete: Course 1 - Module 1-2",
                "Complete: Course 1 - Module 3-4",
                "LeetCode: Excel Sheet Column Number (#171)",
                "LeetCode: Happy Number (#202)"
              ]
            },
            {
              id: "mar-26-w4",
              title: "Week 4: Mar 22-28",
              focus: "Google Data Analytics Cert (Week 4-6)",
              hours: 7,
              depth: "Certificate",
              depthCheck: "N/A - Complete capstone",
              reference: "coursera.org/professional-certificates/google-data-analytics",
              tasks: [
                "Complete: Course 1 - Module 5-6",
                "Complete: Course 2 - Week 1-2",
                "LeetCode: Reverse Linked List (#206)",
                "LeetCode: Contains Duplicate II (#219)",
                "Milestone: ~30 LeetCode problems solved"
              ]
            }
          ]
        }
      ]
    },
    {
      id: "phase-2",
      title: "Phase 2: ML Fundamentals",
      subtitle: "Apr 2026 - Jul 2026 (1hr/day = 7hr/week)",
      months: [
        {
          id: "apr-2026",
          name: "April 2026",
          weeks: [
            {
              id: "apr-26-w1",
              title: "Week 1: Apr 1-7",
              focus: "Linear Regression Deep Dive",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you derive the gradient update rule for linear regression by hand?",
              reference: "Andrew Ng ML Specialization (Coursera)",
              tasks: [
                "Enroll: Andrew Ng ML Specialization",
                "Watch: Week 1 - Supervised ML Intro",
                "Code: Implement linear regression from scratch (NumPy)",
                "LeetCode: Invert Binary Tree (#226)",
                "LeetCode: Summary Ranges (#228)"
              ]
            },
            {
              id: "apr-26-w2",
              title: "Week 2: Apr 8-14",
              focus: "Gradient Descent",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you explain learning rate effects and implement batch vs stochastic GD?",
              reference: "Andrew Ng ML Specialization",
              tasks: [
                "Watch: Week 2 - Gradient Descent",
                "Code: Implement gradient descent with different learning rates",
                "Visualize: Plot cost function convergence",
                "LeetCode: Power of Two (#231)",
                "LeetCode: Palindrome Linked List (#234)",
                "LeetCode: Valid Anagram (#242)"
              ]
            },
            {
              id: "apr-26-w3",
              title: "Week 3: Apr 15-21",
              focus: "Logistic Regression",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you derive cross-entropy loss and explain sigmoid saturation?",
              reference: "Andrew Ng ML Specialization",
              tasks: [
                "Watch: Logistic Regression module",
                "Code: Implement logistic regression from scratch",
                "Code: Binary classification on Iris dataset",
                "LeetCode: Missing Number (#268)",
                "LeetCode: Move Zeroes (#283)",
                "LeetCode: Word Pattern (#290)"
              ]
            },
            {
              id: "apr-26-w4",
              title: "Week 4: Apr 22-28",
              focus: "Model Evaluation",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "When do you use Precision vs Recall vs F1? What's AUC-ROC?",
              reference: "scikit-learn.org/stable/modules/model_evaluation.html",
              tasks: [
                "Read: Scikit-learn - Model Evaluation docs",
                "Code: Compute confusion matrix, precision, recall, F1 by hand",
                "Code: Plot ROC curve for your logistic regression",
                "LeetCode: First Bad Version (#278)",
                "LeetCode: Nim Game (#292)",
                "Claim: Google Data Analytics Certificate (if complete)"
              ]
            }
          ]
        },
        {
          id: "may-2026",
          name: "May 2026",
          weeks: [
            {
              id: "may-26-w1",
              title: "Week 1: May 1-7",
              focus: "Neural Networks Intro",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you compute forward pass for a 2-layer network by hand?",
              reference: "deeplearningbook.org Ch 6",
              tasks: [
                "Watch: Andrew Ng - Neural Networks basics",
                "Read: Deep Learning Book Ch 6 (6.1-6.3)",
                "Code: Implement perceptron from scratch",
                "LeetCode: Range Sum Query (#303)",
                "LeetCode: Intersection of Arrays II (#350)"
              ]
            },
            {
              id: "may-26-w2",
              title: "Week 2: May 8-14",
              focus: "Backpropagation",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Walk through backprop in a 2-layer network with ReLU - this WILL be asked in interviews!",
              reference: "cs231n.github.io/optimization-2/",
              tasks: [
                "Watch: Andrew Ng - Backpropagation",
                "Read: CS231n - Backprop notes",
                "Code: Implement backprop for 2-layer network (NumPy only)",
                "LeetCode: Guess Number (#374)",
                "LeetCode: Ransom Note (#383)",
                "Depth Check: Derive gradients on paper"
              ]
            },
            {
              id: "may-26-w3",
              title: "Week 3: May 15-21",
              focus: "Activation Functions & Optimizers",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Explain ReLU vs Sigmoid pros/cons. What does Adam optimizer do differently?",
              reference: "cs231n.github.io/neural-networks-1/",
              tasks: [
                "Read: CS231n - Activations (ReLU, Sigmoid, Tanh)",
                "Read: CS231n - SGD, Momentum, Adam",
                "Code: Implement Adam optimizer from paper",
                "LeetCode: First Unique Character (#387)",
                "LeetCode: Find the Difference (#389)",
                "LeetCode: Is Subsequence (#392)"
              ]
            },
            {
              id: "may-26-w4",
              title: "Week 4: May 22-28",
              focus: "Complete Andrew Ng Course 1",
              hours: 7,
              depth: "Certificate",
              depthCheck: "N/A - Complete assignments",
              reference: "Andrew Ng ML Specialization",
              tasks: [
                "Complete: All Course 1 assignments",
                "Complete: Course 1 final quiz",
                "LeetCode: Sum of Two Integers (#371)",
                "LeetCode: Nth Digit (#400)",
                "Milestone: Course 1 (Supervised ML) complete",
                "Milestone: ~55 LeetCode problems"
              ]
            }
          ]
        },
        {
          id: "jun-2026",
          name: "June 2026",
          weeks: [
            {
              id: "jun-26-w1",
              title: "Week 1: Jun 1-7",
              focus: "PyTorch Tensors & Autograd",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Explain requires_grad, backward(), and gradient accumulation",
              reference: "pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html",
              tasks: [
                "Read: PyTorch - Tensors Tutorial",
                "Code: Create tensors, move to GPU, basic operations",
                "Read: PyTorch - Autograd Tutorial",
                "Code: Compute gradients manually vs autograd",
                "LeetCode: Add Strings (#415)",
                "LeetCode: Number of Segments (#434)"
              ]
            },
            {
              id: "jun-26-w2",
              title: "Week 2: Jun 8-14",
              focus: "PyTorch nn.Module",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you write a custom nn.Module with forward() and explain parameters()?",
              reference: "pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
              tasks: [
                "Read: PyTorch - Build Model Tutorial",
                "Code: Create custom 3-layer MLP as nn.Module",
                "Code: Training loop (zero_grad, backward, step)",
                "LeetCode: Arranging Coins (#441)",
                "LeetCode: Find All Anagrams (#438)",
                "LeetCode: Minimum Moves (#453)"
              ]
            },
            {
              id: "jun-26-w3",
              title: "Week 3: Jun 15-21",
              focus: "PyTorch DataLoader & Training",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Explain custom Dataset class, DataLoader params (batch_size, shuffle, num_workers)",
              reference: "pytorch.org/tutorials/beginner/basics/data_tutorial.html",
              tasks: [
                "Read: PyTorch - Data Tutorial",
                "Code: Custom Dataset class for your data",
                "Code: Complete training pipeline with validation",
                "LeetCode: Assign Cookies (#455)",
                "LeetCode: Repeated Substring (#459)",
                "LeetCode: Hamming Distance (#461)"
              ]
            },
            {
              id: "jun-26-w4",
              title: "Week 4: Jun 22-28",
              focus: "MNIST Classifier Project",
              hours: 7,
              depth: "Project",
              depthCheck: "Can you build, train, and evaluate an MLP on MNIST from scratch?",
              reference: "pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html",
              tasks: [
                "Code: Build MLP classifier for MNIST",
                "Code: Train and log metrics",
                "Code: Evaluate accuracy, plot confusion matrix",
                "GitHub: Push MNIST project with README",
                "LeetCode: Island Perimeter (#463)",
                "LeetCode: Number Complement (#476)",
                "Milestone: First PyTorch project complete"
              ]
            }
          ]
        },
        {
          id: "jul-2026",
          name: "July 2026",
          weeks: [
            {
              id: "jul-26-w1",
              title: "Week 1: Jul 1-7",
              focus: "CNN Theory",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Given 224×224×3 input, 3×3 conv with 64 filters, stride 1, pad 1: what's output shape?",
              reference: "cs231n.github.io/convolutional-networks/",
              tasks: [
                "Read: CS231n - CNN notes (complete)",
                "Practice: Compute output shapes for 5 different conv configs",
                "Watch: 3Blue1Brown - Convolutions",
                "LeetCode: License Key Formatting (#482)",
                "LeetCode: Max Consecutive Ones (#485)"
              ]
            },
            {
              id: "jul-26-w2",
              title: "Week 2: Jul 8-14",
              focus: "CNN Architectures",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "What problem does ResNet solve? Explain skip connections.",
              reference: "Read ResNet paper introduction & method",
              tasks: [
                "Read: LeNet, AlexNet architectures (skim)",
                "Read: VGG architecture",
                "Read: ResNet paper (focus on skip connections)",
                "Code: Build simple CNN in PyTorch",
                "LeetCode: Teemo Attacking (#495)",
                "LeetCode: Relative Ranks (#506)"
              ]
            },
            {
              id: "jul-26-w3",
              title: "Week 3: Jul 15-21",
              focus: "Transfer Learning",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "When do you freeze layers? When to fine-tune all? What's learning rate choice?",
              reference: "pytorch.org/tutorials/beginner/transfer_learning_tutorial.html",
              tasks: [
                "Read: PyTorch Transfer Learning Tutorial",
                "Code: Load pretrained ResNet, freeze layers",
                "Code: Fine-tune on custom dataset (10 classes)",
                "LeetCode: Detect Capital (#520)",
                "LeetCode: Longest Uncommon Subsequence (#521)",
                "LeetCode: Keyboard Row (#500)"
              ]
            },
            {
              id: "jul-26-w4",
              title: "Week 4: Jul 22-28",
              focus: "CNN Project - Image Classifier",
              hours: 7,
              depth: "Project",
              depthCheck: "Full CNN project with data augmentation, training, evaluation",
              reference: "Deploy on HuggingFace Spaces",
              tasks: [
                "Code: Image Classifier with data augmentation",
                "Code: Train on custom dataset",
                "Deploy: Push to HuggingFace Spaces",
                "GitHub: Complete README with demo link",
                "LeetCode: Coin Change II (#518)",
                "Milestone: ~80 LeetCode problems",
                "Claim: DeepLearning.AI ML Specialization (if complete)"
              ]
            }
          ]
        }
      ]
    },
    {
      id: "phase-3",
      title: "Phase 3: Deep Learning & Specialization",
      subtitle: "Aug 2026 - Dec 2026 (1hr/day + 2hrs weekend for CUDA/Kaggle)",
      months: [
        {
          id: "aug-2026",
          name: "August 2026",
          weeks: [
            {
              id: "aug-26-w1",
              title: "Week 1: Aug 1-7",
              focus: "LeetCode Grind - Arrays & Hashing",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you solve Medium array problems in 20-30 mins without hints?",
              reference: "neetcode.io/roadmap",
              tasks: [
                "Solve: Product of Array Except Self (#238)",
                "Solve: Top K Frequent Elements (#347)",
                "Solve: Encode and Decode Strings",
                "Solve: Valid Sudoku (#36)",
                "Solve: Longest Consecutive Sequence (#128)"
              ]
            },
            {
              id: "aug-26-w2",
              title: "Week 2: Aug 8-14",
              focus: "LeetCode Grind - Two Pointers & Sliding Window",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Recognize when to use sliding window vs two pointers pattern",
              reference: "neetcode.io/roadmap",
              tasks: [
                "Solve: 3Sum (#15)",
                "Solve: Container With Most Water (#11)",
                "Solve: Trapping Rain Water (#42)",
                "Solve: Longest Substring Without Repeat (#3)",
                "Solve: Minimum Window Substring (#76)"
              ]
            },
            {
              id: "aug-26-w3",
              title: "Week 3: Aug 15-21",
              focus: "LeetCode Grind - Binary Search & Trees",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "Can you implement binary search variants (first, last, rotated)?",
              reference: "neetcode.io/roadmap",
              tasks: [
                "Solve: Search in Rotated Sorted Array (#33)",
                "Solve: Find Minimum in Rotated Sorted (#153)",
                "Solve: Validate BST (#98)",
                "Solve: Kth Smallest in BST (#230)",
                "Solve: LCA of BST (#235)"
              ]
            },
            {
              id: "aug-26-w4",
              title: "Week 4: Aug 22-28",
              focus: "LeetCode Grind - DFS & BFS",
              hours: 7,
              depth: "🔴 Expert",
              depthCheck: "When to use DFS vs BFS? Implement both iteratively and recursively",
              reference: "neetcode.io/roadmap",
              tasks: [
                "Solve: Number of Islands (#200)",
                "Solve: Clone Graph (#133)",
                "Solve: Pacific Atlantic Water Flow (#417)",
                "Solve: Course Schedule (#207)",
                "Solve: Course Schedule II (#210)",
                "Milestone: 100+ LeetCode problems"
              ]
            }
          ]
        },
        {
          id: "sep-2026",
          name: "September 2026",
          weeks: [
            {
              id: "sep-26-w1",
              title: "Week 1: Sep 1-7",
              focus: "RNNs & Sequences",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Explain vanishing gradients in RNNs and how LSTM gates solve it",
              reference: "colah.github.io/posts/2015-08-Understanding-LSTMs/",
              tasks: [
                "Read: Colah - Understanding LSTMs (complete)",
                "Code: Simple RNN in PyTorch",
                "Code: LSTM for sequence classification",
                "LeetCode: Graph Valid Tree (#261)",
                "LeetCode: Walls and Gates (#286)"
              ]
            },
            {
              id: "sep-26-w2",
              title: "Week 2: Sep 8-14",
              focus: "NLP Basics - Text Processing",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Explain tokenization, word embeddings, and why we need padding",
              reference: "pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html",
              tasks: [
                "Read: PyTorch Text Classification Tutorial",
                "Code: Tokenization, vocabulary building",
                "Code: Word embeddings (nn.Embedding)",
                "LeetCode: Meeting Rooms (#252)",
                "LeetCode: Meeting Rooms II (#253)"
              ]
            },
            {
              id: "sep-26-w3",
              title: "Week 3: Sep 15-21",
              focus: "Sentiment Analysis Project",
              hours: 7,
              depth: "Project",
              depthCheck: "Full NLP pipeline: preprocess → embed → model → evaluate",
              reference: "Use IMDB or Twitter sentiment dataset",
              tasks: [
                "Code: Load and preprocess sentiment dataset",
                "Code: Build LSTM classifier",
                "Code: Train and evaluate",
                "LeetCode: Alien Dictionary (#269)",
                "LeetCode: Serialize Binary Tree (#297)"
              ]
            },
            {
              id: "sep-26-w4",
              title: "Week 4: Sep 22-28",
              focus: "Complete NLP Project",
              hours: 7,
              depth: "Project",
              depthCheck: "Polish and deploy",
              reference: "GitHub + README",
              tasks: [
                "Code: Add attention mechanism to LSTM",
                "Code: Hyperparameter tuning",
                "GitHub: Push with comprehensive README",
                "LeetCode: Word Search II (#212)",
                "LeetCode: House Robber III (#337)",
                "Milestone: Sentiment Analysis project complete"
              ]
            }
          ]
        },
        {
          id: "oct-2026",
          name: "October 2026",
          weeks: [
            {
              id: "oct-26-w1",
              title: "Week 1: Oct 1-7",
              focus: "Attention Mechanism + CUDA Intro (Weekend)",
              hours: 9,
              depth: "🔴 Expert (Attention) + 🟢 Awareness (CUDA)",
              depthCheck: "Compute attention(Q,K,V) by hand. Explain why GPUs beat CPUs for DL.",
              reference: "jalammar.github.io/illustrated-transformer/ + docs.nvidia.com/cuda/",
              tasks: [
                "Read: Illustrated Transformer (Part 1 - Attention)",
                "Code: Implement scaled dot-product attention from scratch",
                "Practice: Trace attention computation by hand",
                "Weekend (2hrs): Read CUDA intro - threads vs blocks concept",
                "Weekend: Understand why GPUs parallelize better than CPUs",
                "LeetCode: Coin Change (#322)",
                "LeetCode: Longest Increasing Subsequence (#300)"
              ]
            },
            {
              id: "oct-26-w2",
              title: "Week 2: Oct 8-14",
              focus: "Self-Attention & Multi-Head + CUDA Memory (Weekend)",
              hours: 9,
              depth: "🔴 Expert (Attention) + 🟢 Awareness (CUDA)",
              depthCheck: "Why multiple heads? Explain global vs shared memory in CUDA.",
              reference: "jalammar.github.io/illustrated-transformer/ + CUDA docs",
              tasks: [
                "Read: Illustrated Transformer (Part 2 - Multi-Head)",
                "Code: Implement multi-head attention",
                "Code: Visualize attention weights",
                "Weekend (2hrs): CUDA memory hierarchy (global, shared, local)",
                "Weekend: Understand how PyTorch uses CUDA under the hood",
                "LeetCode: Word Break (#139)",
                "LeetCode: Decode Ways (#91)"
              ]
            },
            {
              id: "oct-26-w3",
              title: "Week 3: Oct 15-21",
              focus: "Full Transformer Architecture + Kaggle Setup (Weekend)",
              hours: 9,
              depth: "🟡 Working (Transformer) + Kaggle Prep",
              depthCheck: "Explain encoder vs decoder stack, positional encoding, layer norm placement",
              reference: "nlp.seas.harvard.edu/2018/04/03/attention.html + Kaggle",
              tasks: [
                "Read: Annotated Transformer (complete)",
                "Code: Implement transformer encoder block",
                "Code: Add positional encoding",
                "Weekend (2hrs): Pick ONE image-based Kaggle competition",
                "Weekend: Set up experiment tracking notebook",
                "LeetCode: Unique Paths (#62)",
                "LeetCode: Jump Game (#55)"
              ]
            },
            {
              id: "oct-26-w4",
              title: "Week 4: Oct 22-28",
              focus: "BERT & HuggingFace + Kaggle Baseline (Weekend)",
              hours: 9,
              depth: "🟡 Working (BERT) + Kaggle Project",
              depthCheck: "Explain BERT's pre-training tasks (MLM, NSP) and fine-tuning process",
              reference: "huggingface.co/docs/transformers/ + Kaggle",
              tasks: [
                "Read: HuggingFace Transformers quick tour",
                "Code: Load pre-trained BERT",
                "Code: Fine-tune BERT for text classification",
                "Weekend (2hrs): Kaggle - Build baseline CNN model",
                "Weekend: Log first experiments with metrics",
                "LeetCode: House Robber (#198)",
                "LeetCode: House Robber II (#213)",
                "Milestone: 125+ LeetCode"
              ]
            }
          ]
        },
        {
          id: "nov-2026",
          name: "November 2026",
          weeks: [
            {
              id: "nov-26-w1",
              title: "Week 1: Nov 1-7",
              focus: "NVIDIA DL Certificate - Day 1-2 + CUDA Review (Weekend)",
              hours: 18,
              depth: "Certificate + 🟢 CUDA Awareness",
              depthCheck: "N/A - Take leave, complete intensive course + review CUDA basics",
              reference: "nvidia.com/en-us/training/ + CUDA Programming Guide",
              tasks: [
                "Take sick leave from internship",
                "NVIDIA DLI: Module 1-2",
                "NVIDIA DLI: Module 3-4",
                "Weekend (2hrs): Review CUDA concepts from Oct",
                "Weekend: Connect CUDA knowledge to NVIDIA DLI content"
              ]
            },
            {
              id: "nov-26-w2",
              title: "Week 2: Nov 8-14",
              focus: "NVIDIA DL Certificate - Complete + Kaggle Experiments (Weekend)",
              hours: 26,
              depth: "Certificate + Kaggle Work",
              depthCheck: "N/A - Complete cert and run Kaggle experiments",
              reference: "nvidia.com/en-us/training/ + Kaggle",
              tasks: [
                "NVIDIA DLI: Remaining modules",
                "NVIDIA DLI: Final assessment",
                "Claim: NVIDIA Fundamentals of Deep Learning Certificate",
                "Weekend (2hrs): Kaggle - Try ResNet architecture",
                "Weekend: Compare CNN vs ResNet performance"
              ]
            },
            {
              id: "nov-26-w3",
              title: "Week 3: Nov 15-21",
              focus: "Dynamic Programming LeetCode + Kaggle Error Analysis (Weekend)",
              hours: 9,
              depth: "🔴 Expert (DP) + Kaggle Analysis",
              depthCheck: "Recognize DP patterns: 1D, 2D, LCS, knapsack variants",
              reference: "neetcode.io/roadmap + Kaggle notebook",
              tasks: [
                "Solve: Climbing Stairs (#70)",
                "Solve: Coin Change (#322)",
                "Solve: Longest Common Subsequence (#1143)",
                "Solve: Edit Distance (#72)",
                "Solve: Distinct Subsequences (#115)",
                "Weekend (2hrs): Kaggle - Error analysis on failed predictions",
                "Weekend: Document experiment insights"
              ]
            },
            {
              id: "nov-26-w4",
              title: "Week 4: Nov 22-28",
              focus: "More DP + Graphs + Kaggle Wrap-up (Weekend)",
              hours: 9,
              depth: "🔴 Expert (DSA) + Kaggle Final",
              depthCheck: "Can you solve 2D DP problems and graph traversals confidently?",
              reference: "neetcode.io/roadmap + Kaggle",
              tasks: [
                "Solve: Maximum Product Subarray (#152)",
                "Solve: Partition Equal Subset Sum (#416)",
                "Solve: Network Delay Time (#743)",
                "Solve: Cheapest Flights K Stops (#787)",
                "Weekend (2hrs): Kaggle - Finalize submission",
                "Weekend: Write Kaggle notebook summary with learnings",
                "Milestone: 145+ LeetCode"
              ]
            }
          ]
        },
        {
          id: "dec-2026",
          name: "December 2026",
          weeks: [
            {
              id: "dec-26-w1",
              title: "Week 1: Dec 1-7",
              focus: "Transformer Project - Fine-tune BERT + Kaggle Documentation (Weekend)",
              hours: 9,
              depth: "Project + Kaggle Wrap-up",
              depthCheck: "Full pipeline: load pretrained → fine-tune → evaluate → deploy",
              reference: "HuggingFace Hub + Kaggle notebook",
              tasks: [
                "Code: Pick text classification task",
                "Code: Load and preprocess dataset",
                "Code: Fine-tune BERT/DistilBERT",
                "Weekend (2hrs): Kaggle - Write comprehensive notebook summary",
                "Weekend: Document all experiments and learnings",
                "LeetCode: Swim in Rising Water (#778)",
                "LeetCode: Reconstruct Itinerary (#332)"
              ]
            },
            {
              id: "dec-26-w2",
              title: "Week 2: Dec 8-14",
              focus: "Transformer Project Complete",
              hours: 7,
              depth: "Project",
              depthCheck: "Deployed transformer project with metrics",
              reference: "Deploy to HuggingFace Spaces",
              tasks: [
                "Code: Evaluate model, compute metrics",
                "Code: Create inference pipeline",
                "Deploy: HuggingFace Spaces",
                "GitHub: Push with README",
                "Milestone: Transformer project complete"
              ]
            },
            {
              id: "dec-26-w3",
              title: "Week 3: Dec 15-21",
              focus: "AWS ML Foundations Start",
              hours: 7,
              depth: "Certificate",
              depthCheck: "N/A - Complete Udacity course",
              reference: "udacity.com/course/aws-machine-learning-foundations",
              tasks: [
                "Enroll: AWS ML Foundations (Udacity - Free)",
                "Complete: Module 1-2",
                "Complete: Module 3",
                "LeetCode: Min Cost Climbing Stairs (#746)",
                "LeetCode: Delete and Earn (#740)"
              ]
            },
            {
              id: "dec-26-w4",
              title: "Week 4: Dec 22-28",
              focus: "GitHub Portfolio Cleanup",
              hours: 7,
              depth: "Portfolio",
              depthCheck: "5-6 polished projects with excellent READMEs",
              reference: "All project repos",
              tasks: [
                "Review: All GitHub repos",
                "Delete: Weak/incomplete projects",
                "Polish: READMEs for best 5-6 projects",
                "LeetCode: Ones and Zeroes (#474)",
                "Milestone: 160+ LeetCode, Clean portfolio"
              ]
            }
          ]
        }
      ]
    },
    {
      id: "phase-4",
      title: "Phase 4: Final Push",
      subtitle: "Jan 2027 - Apr 2027 (1hr/day + 2hrs weekend for C++/Systems)",
      months: [
        {
          id: "jan-2027",
          name: "January 2027",
          weeks: [
            {
              id: "jan-27-w1",
              title: "Week 1: Jan 1-7",
              focus: "AWS ML Cert Complete",
              hours: 7,
              depth: "Certificate",
              depthCheck: "N/A - Complete and claim",
              reference: "Udacity AWS ML Foundations",
              tasks: [
                "Complete: AWS ML remaining modules",
                "Complete: Final assessment",
                "Claim: AWS ML Foundations Certificate",
                "LeetCode: Profitable Schemes (#879)",
                "LeetCode: Last Stone Weight II (#1049)"
              ]
            },
            {
              id: "jan-27-w2",
              title: "Week 2: Jan 8-14",
              focus: "Multi-Model Comparison Project - Plan",
              hours: 7,
              depth: "Project",
              depthCheck: "Compare CNN vs ResNet vs ViT on same dataset",
              reference: "Pick image classification dataset",
              tasks: [
                "Plan: Select dataset, define metrics",
                "Code: Data pipeline setup",
                "Code: CNN baseline model",
                "LeetCode: Target Sum (#494)",
                "LeetCode: Find Min in Rotated Sorted II (#154)"
              ]
            },
            {
              id: "jan-27-w3",
              title: "Week 3: Jan 15-21",
              focus: "Multi-Model Project - ResNet",
              hours: 7,
              depth: "Project",
              depthCheck: "Train and evaluate ResNet, understand skip connections in practice",
              reference: "torchvision.models.resnet",
              tasks: [
                "Code: ResNet (pretrained + fine-tune)",
                "Code: Train and log metrics",
                "Code: Compare with CNN baseline",
                "LeetCode: Redundant Connection (#684)",
                "LeetCode: Max Area of Island (#695)"
              ]
            },
            {
              id: "jan-27-w4",
              title: "Week 4: Jan 22-28",
              focus: "Multi-Model Project - ViT",
              hours: 7,
              depth: "Project",
              depthCheck: "Understand Vision Transformer and patch embeddings",
              reference: "huggingface.co/docs/transformers/model_doc/vit",
              tasks: [
                "Read: ViT paper introduction",
                "Code: ViT (pretrained + fine-tune)",
                "Code: Benchmark all 3 models",
                "LeetCode: Accounts Merge (#721)",
                "LeetCode: Rotting Oranges (#994)",
                "Milestone: 175+ LeetCode"
              ]
            }
          ]
        },
        {
          id: "feb-2027",
          name: "February 2027",
          weeks: [
            {
              id: "feb-27-w1",
              title: "Week 1: Feb 1-7",
              focus: "Multi-Model Project Complete + C++ Intro (Weekend)",
              hours: 9,
              depth: "Project + 🟢 C++ Awareness",
              depthCheck: "Full analysis with charts, metrics, conclusions",
              reference: "GitHub final push + learncpp.com",
              tasks: [
                "Code: Create comparison charts/tables",
                "Code: Write detailed analysis",
                "GitHub: Push with comprehensive README",
                "Weekend (2hrs): C++ basics - variables, types, syntax",
                "Weekend: Understand pointers and memory concepts",
                "Milestone: 5th major project complete"
              ]
            },
            {
              id: "feb-27-w2",
              title: "Week 2: Feb 8-14",
              focus: "ML System Design Intro + C++ Memory (Weekend)",
              hours: 9,
              depth: "🟡 Working (System Design) + 🟢 C++",
              depthCheck: "Explain end-to-end ML system: data → train → serve → monitor",
              reference: "Chip Huyen - Designing ML Systems + C++ tutorials",
              tasks: [
                "Read: Chip Huyen book - Chapter 1-2",
                "Practice: Design a simple RecSys on paper",
                "Weekend (2hrs): C++ memory - stack vs heap",
                "Weekend: Understand references vs pointers",
                "LeetCode: Surrounded Regions (#130)",
                "LeetCode: Word Ladder (#127)"
              ]
            },
            {
              id: "feb-27-w3",
              title: "Week 3: Feb 15-21",
              focus: "ML System Design Practice + PyTorch Internals (Weekend)",
              hours: 9,
              depth: "🟡 Working (System Design) + 🟢 PyTorch",
              depthCheck: "Can you design image search or recommendation system?",
              reference: "github.com/chiphuyen/machine-learning-systems-design + PyTorch docs",
              tasks: [
                "Read: ML System Design GitHub guide",
                "Practice: Design image search system",
                "Practice: Design fraud detection system",
                "Weekend (2hrs): How PyTorch uses C++ backends (ATen)",
                "Weekend: Explore PyTorch C++ extension basics",
                "LeetCode: Minimum Height Trees (#310)",
                "LeetCode: Evaluate Division (#399)"
              ]
            },
            {
              id: "feb-27-w4",
              title: "Week 4: Feb 22-28",
              focus: "Hard LeetCode Prep + PyTorch CUDA Connection (Weekend)",
              hours: 9,
              depth: "🔴 Expert (LC) + 🟢 Systems",
              depthCheck: "Can you approach Hard problems systematically?",
              reference: "NeetCode Hard playlist + PyTorch docs",
              tasks: [
                "Solve: Median of Two Sorted Arrays (#4)",
                "Solve: Merge K Sorted Lists (#23)",
                "Solve: Trapping Rain Water (#42)",
                "Solve: N-Queens (#51)",
                "Weekend (2hrs): How PyTorch tensors connect to CUDA",
                "Weekend: Understand .to('cuda') under the hood",
                "Milestone: 185+ LeetCode"
              ]
            }
          ]
        },
        {
          id: "mar-2027",
          name: "March 2027",
          weeks: [
            {
              id: "mar-27-w1",
              title: "Week 1: Mar 1-7",
              focus: "More Hard LeetCode + C++ Review (Weekend)",
              hours: 9,
              depth: "🔴 Expert (LC) + 🟢 C++ Review",
              depthCheck: "Solve Hard problems in 45-60 mins",
              reference: "NeetCode Hard playlist + C++ notes",
              tasks: [
                "Solve: Longest Valid Parentheses (#32)",
                "Solve: Wildcard Matching (#44)",
                "Solve: Word Break II (#140)",
                "Solve: Burst Balloons (#312)",
                "Weekend (2hrs): Review all C++ concepts from Feb",
                "Weekend: Practice explaining memory management"
              ]
            },
            {
              id: "mar-27-w2",
              title: "Week 2: Mar 8-14",
              focus: "Mock Interviews - Technical + Systems Review (Weekend)",
              hours: 9,
              depth: "Interview Prep + 🟢 Systems",
              depthCheck: "Communicate solution clearly while coding",
              reference: "pramp.com or interviewing.io + CUDA/C++ notes",
              tasks: [
                "Complete: 2 Pramp mock interviews",
                "Review: Feedback and improve",
                "Weekend (2hrs): Review CUDA concepts from Oct-Nov",
                "Weekend: Prepare to answer 'Why NVIDIA?' with CUDA knowledge",
                "LeetCode: Regular Expression Matching (#10)",
                "LeetCode: Edit Distance (#72)"
              ]
            },
            {
              id: "mar-27-w3",
              title: "Week 3: Mar 15-21",
              focus: "Mock Interviews - ML/System Design",
              hours: 7,
              depth: "Interview Prep",
              depthCheck: "Explain your ML projects confidently",
              reference: "Prepare project explanations",
              tasks: [
                "Practice: Walk through each project (5 min each)",
                "Practice: Answer 'Why did you choose X?' questions",
                "Practice: System design (recommendation system)",
                "LeetCode: Serialize Deserialize Binary Tree (#297)",
                "LeetCode: Find Median Data Stream (#295)"
              ]
            },
            {
              id: "mar-27-w4",
              title: "Week 4: Mar 22-28",
              focus: "Behavioral Prep (STAR Method)",
              hours: 7,
              depth: "Interview Prep",
              depthCheck: "5 polished STAR stories ready",
              reference: "Prepare stories using STAR format",
              tasks: [
                "Prepare: 'Tell me about yourself' (90 sec)",
                "Prepare: 'Challenging project' story",
                "Prepare: 'Team conflict' story",
                "Prepare: 'Why NVIDIA?' answer",
                "Milestone: 200+ LeetCode, Interview-ready"
              ]
            }
          ]
        },
        {
          id: "apr-2027",
          name: "April 2027",
          weeks: [
            {
              id: "apr-27-w1",
              title: "Week 1: Apr 1-7",
              focus: "TensorFlow Crash Course",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Can you build and train models in TensorFlow (not just PyTorch)?",
              reference: "tensorflow.org/tutorials/quickstart/beginner",
              tasks: [
                "Read: TensorFlow Quickstart",
                "Code: Keras Sequential model",
                "Code: MNIST in TensorFlow",
                "Practice: TF Developer Cert prep questions"
              ]
            },
            {
              id: "apr-27-w2",
              title: "Week 2: Apr 8-14",
              focus: "TensorFlow Advanced",
              hours: 7,
              depth: "🟡 Working",
              depthCheck: "Custom training loops, tf.data, callbacks",
              reference: "tensorflow.org/guide",
              tasks: [
                "Code: Custom training loop",
                "Code: tf.data input pipeline",
                "Code: Model callbacks (early stopping, checkpoints)",
                "Practice: More cert prep"
              ]
            },
            {
              id: "apr-27-w3",
              title: "Week 3: Apr 15-21",
              focus: "Resume Final Polish",
              hours: 7,
              depth: "Career Prep",
              depthCheck: "1-page resume with 5 certs, 5 projects, 200+ LC",
              reference: "r/cscareerquestions for reviews",
              tasks: [
                "Update: Resume with all achievements",
                "Review: Get feedback from r/cscareerquestions",
                "Fix: All formatting issues",
                "Optimize: ATS-friendly format"
              ]
            },
            {
              id: "apr-27-w4",
              title: "Week 4: Apr 22-28",
              focus: "LinkedIn + Prep to Quit",
              hours: 7,
              depth: "Career Prep",
              depthCheck: "LinkedIn optimized, ready for full grind",
              reference: "LinkedIn profile",
              tasks: [
                "Update: LinkedIn headline, about, experience",
                "Add: All certifications to LinkedIn",
                "Add: Project links to featured section",
                "Plan: Quit internship end of month",
                "Milestone: Ready for Phase 5 Full Grind"
              ]
            }
          ]
        }
      ]
    },
    {
      id: "phase-5",
      title: "Phase 5: Full Grind Mode",
      subtitle: "May 2027 - Jul 2027 (No Job = Full Time Prep)",
      months: [
        {
          id: "may-2027",
          name: "May 2027",
          weeks: [
            {
              id: "may-27-w1",
              title: "Week 1: May 1-7",
              focus: "TensorFlow Developer Cert - Intensive",
              hours: 40,
              depth: "Certificate",
              depthCheck: "N/A - Full-time cert prep",
              reference: "tensorflow.org/certificate",
              tasks: [
                "Study: TF Developer Cert curriculum (full time)",
                "Practice: Image classification models",
                "Practice: NLP models",
                "Practice: Time series models"
              ]
            },
            {
              id: "may-27-w2",
              title: "Week 2: May 8-14",
              focus: "TensorFlow Cert Exam",
              hours: 40,
              depth: "Certificate",
              depthCheck: "N/A - Take and pass exam",
              reference: "tensorflow.org/certificate",
              tasks: [
                "Review: All practice problems",
                "Take: TensorFlow Developer Certification Exam",
                "Claim: TensorFlow Developer Certificate",
                "Milestone: ALL 5 CERTIFICATIONS COMPLETE"
              ]
            },
            {
              id: "may-27-w3",
              title: "Week 3: May 15-21",
              focus: "Full-Stack ML App - Backend",
              hours: 40,
              depth: "Project",
              depthCheck: "FastAPI serving ML model with proper endpoints",
              reference: "fastapi.tiangolo.com",
              tasks: [
                "Code: FastAPI app setup",
                "Code: ML model inference endpoint",
                "Code: Input validation, error handling",
                "Code: Dockerize backend"
              ]
            },
            {
              id: "may-27-w4",
              title: "Week 4: May 22-28",
              focus: "Full-Stack ML App - Frontend & Deploy",
              hours: 40,
              depth: "Project",
              depthCheck: "Deployed full-stack ML app",
              reference: "Deploy on Render/Railway",
              tasks: [
                "Code: Simple frontend (React/HTML)",
                "Code: Connect to backend",
                "Deploy: Full app on Render or Railway",
                "GitHub: Polish README with demo",
                "Milestone: Full-stack deployed project"
              ]
            }
          ]
        },
        {
          id: "jun-2027",
          name: "June 2027",
          weeks: [
            {
              id: "jun-27-w1",
              title: "Week 1: Jun 1-7",
              focus: "Hard LeetCode Blitz",
              hours: 40,
              depth: "🔴 Expert",
              depthCheck: "Solve 5 Hard problems per day",
              reference: "LeetCode Premium (company tags)",
              tasks: [
                "Solve: 15 Hard problems",
                "Focus: DP, Graphs, Trees",
                "Review: All previous Medium problems",
                "Goal: 215+ LeetCode"
              ]
            },
            {
              id: "jun-27-w2",
              title: "Week 2: Jun 8-14",
              focus: "NVIDIA-Specific Prep",
              hours: 40,
              depth: "🟢 Awareness",
              depthCheck: "Understand CUDA concepts, TensorRT, NVIDIA's tech stack",
              reference: "docs.nvidia.com/cuda/",
              tasks: [
                "Read: CUDA Programming Guide (Intro chapters)",
                "Read: TensorRT documentation (concepts)",
                "Watch: NVIDIA GTC talks on YouTube",
                "Solve: NVIDIA-tagged LeetCode problems"
              ]
            },
            {
              id: "jun-27-w3",
              title: "Week 3: Jun 15-21",
              focus: "Daily Mock Interviews",
              hours: 40,
              depth: "Interview Prep",
              depthCheck: "1-2 mock interviews daily",
              reference: "pramp.com, interviewing.io",
              tasks: [
                "Complete: 5 technical mock interviews",
                "Complete: 3 ML system design mocks",
                "Polish: All behavioral answers",
                "Goal: 220+ LeetCode"
              ]
            },
            {
              id: "jun-27-w4",
              title: "Week 4: Jun 22-28",
              focus: "Final Review",
              hours: 40,
              depth: "Review",
              depthCheck: "All concepts fresh, all projects explainable",
              reference: "All previous materials",
              tasks: [
                "Review: All 5 projects (demo-ready)",
                "Review: All 5 certifications knowledge",
                "Review: Top 50 LeetCode problems",
                "Review: System design patterns"
              ]
            }
          ]
        },
        {
          id: "jul-2027",
          name: "July 2027",
          weeks: [
            {
              id: "jul-27-w1",
              title: "Week 1: Jul 1-7",
              focus: "Resume Final Check & Templates",
              hours: 30,
              depth: "Applications",
              depthCheck: "Resume verified, cover letter templates ready",
              reference: "r/cscareerquestions",
              tasks: [
                "Finalize: Resume (get 3 more reviews)",
                "Prepare: Cover letter templates",
                "Create: Company research spreadsheet",
                "List: 50+ target companies"
              ]
            },
            {
              id: "jul-27-w2",
              title: "Week 2: Jul 8-14",
              focus: "Tier 1 Applications",
              hours: 40,
              depth: "Applications",
              depthCheck: "Apply to dream companies",
              reference: "Company career pages",
              tasks: [
                "Apply: NVIDIA (ML/AI roles)",
                "Apply: AMD",
                "Apply: Google",
                "Apply: Microsoft",
                "Apply: Intel",
                "Customize: Each resume/cover letter"
              ]
            },
            {
              id: "jul-27-w3",
              title: "Week 3: Jul 15-21",
              focus: "Tier 2 & 3 Applications",
              hours: 40,
              depth: "Applications",
              depthCheck: "Broad application coverage",
              reference: "Company career pages",
              tasks: [
                "Apply: Amazon, Apple, Meta, Adobe",
                "Apply: Flipkart, PhonePe, Razorpay, CRED",
                "Apply: Qualcomm, Samsung, ARM",
                "Apply: 10+ ML/AI startups",
                "Track: All applications in spreadsheet"
              ]
            },
            {
              id: "jul-27-w4",
              title: "Week 4: Jul 22-28",
              focus: "Continue Applications + Interview Prep",
              hours: 40,
              depth: "Applications",
              depthCheck: "50+ applications, interview-ready",
              reference: "All channels",
              tasks: [
                "Apply: More companies (reach 50+)",
                "Network: LinkedIn connections at target companies",
                "Prep: For interview callbacks",
                "Maintain: LeetCode daily (1-2 problems)",
                "GOAL: 50+ APPLICATIONS SUBMITTED"
              ]
            }
          ]
        }
      ]
    }
  ],
  stats: {
    totalMonths: 18,
    totalCerts: 5,
    totalProjects: 5,
    totalLeetCode: 220
  },
  certifications: [
    { id: "cert-1", name: "Google Data Analytics", month: "Mar-Apr 2026", status: "pending" },
    { id: "cert-2", name: "DeepLearning.AI ML Specialization", month: "Jul 2026", status: "pending" },
    { id: "cert-3", name: "NVIDIA Fundamentals of Deep Learning", month: "Nov 2026", status: "pending" },
    { id: "cert-4", name: "AWS Machine Learning Foundations", month: "Jan 2027", status: "pending" },
    { id: "cert-5", name: "TensorFlow Developer Certificate", month: "May 2027", status: "pending" }
  ],
  projects: [
    { id: "proj-1", name: "MNIST Classifier (PyTorch)", month: "Jun 2026", status: "pending" },
    { id: "proj-2", name: "CNN Image Classifier (Deployed)", month: "Jul 2026", status: "pending" },
    { id: "proj-3", name: "Sentiment Analysis (NLP/LSTM)", month: "Sep 2026", status: "pending" },
    { id: "proj-4", name: "Transformer Fine-tuning (BERT)", month: "Dec 2026", status: "pending" },
    { id: "proj-5", name: "Multi-Model Comparison (CNN vs ResNet vs ViT)", month: "Feb 2027", status: "pending" },
    { id: "proj-6", name: "Full-Stack ML App (FastAPI + Deploy)", month: "May 2027", status: "pending" }
  ]
};
