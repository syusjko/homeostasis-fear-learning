import matplotlib.pyplot as plt
import numpy as np
import os

# Create docs/assets folder if it doesn't exist
os.makedirs('docs/assets', exist_ok=True)

# Graph 1: Conceptual Extinction & Relapse
plt.figure(figsize=(10, 5))
plt.style.use('ggplot')

# Time phases
t_acquisition = np.linspace(0, 10, 50)
t_extinction = np.linspace(10, 20, 50)
t_rest = np.linspace(20, 25, 10)
t_relapse = np.linspace(25, 30, 25)

# Fear response (conceptual)
fear_acq = 1 - np.exp(-t_acquisition)
fear_ext = fear_acq[-1] * np.exp(-(t_extinction - 10))
fear_rest = np.zeros(len(t_rest))
fear_rel = 0.6 * np.exp(-(t_relapse - 25))

t_total = np.concatenate([t_acquisition, t_extinction, t_rest, t_relapse])
fear_total = np.concatenate([fear_acq, fear_ext, fear_rest, fear_rel])

plt.plot(t_total, fear_total, lw=3, color='#e74c3c')
plt.axvline(x=10, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=20, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=25, color='gray', linestyle='--', alpha=0.7)

plt.text(5, 0.5, 'Acquisition', ha='center', fontsize=12)
plt.text(15, 0.5, 'Extinction', ha='center', fontsize=12)
plt.text(22.5, 0.1, 'Rest', ha='center', fontsize=10)
plt.text(27.5, 0.5, 'Relapse', ha='center', fontsize=12)

plt.title('Conceptual Behavioral Signatures: Extinction and Relapse', fontsize=14)
plt.xlabel('Time (Trials)', fontsize=12)
plt.ylabel('Threat/Fear Response Level', fontsize=12)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig('docs/assets/extinction_relapse.png', dpi=150)
plt.close()

# Graph 2: Evolution (Slow) vs Learning (Fast)
plt.figure(figsize=(10, 5))

# Generations
generations = np.arange(0, 20)
# Slow weight (Genetic parameter)
slow_weight = 1 - np.exp(-generations/5)

plt.plot(generations, slow_weight, 'o-', color='#2980b9', lw=2, label='Genotypic Parameter (Slow weight)')

# Fast weights (within lifetime)
for g, sw in zip(generations[::4], slow_weight[::4]):
    t_lifetime = np.linspace(g, g+0.8, 20)
    fast_weight = sw + (1-sw)*(1-np.exp(-(t_lifetime-g)*5))
    plt.plot(t_lifetime, fast_weight, color='#27ae60', lw=1.5, alpha=0.8)
    if g == 0:
        plt.plot(t_lifetime, fast_weight, color='#27ae60', lw=1.5, alpha=0.8, label='Phenotypic Adaptation (Fast weights)')

plt.title('Separation of Scales: Evolution vs. Online Learning', fontsize=14)
plt.xlabel('Evolutionary Time (Generations)', fontsize=12)
plt.ylabel('Trait Value / Synaptic Weight', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('docs/assets/evolution_learning.png', dpi=150)
plt.close()

print("Graphs generated successfully.")
