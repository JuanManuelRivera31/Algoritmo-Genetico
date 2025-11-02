# ga_creature.py
# Proyecto: Algoritmos Genéticos para locomoción con restricciones físicas (proxy 2D)
# Autor: Tú + ChatGPT
# Descripción:
# - Evoluciona una criatura bípedo 2D con control CPG bajo restricciones de masa, altura, torques,
#   rigidez/amortiguamiento (ligamentos), fricción del pie, consumo energético y estabilidad/caída.
# - Este simulador es un "proxy" que simplifica la dinámica, pero conserva las magnitudes y penalizaciones
#   que luego trasladarás 1:1 a Unity (mismos genes, fitness y límites).
#
# Salidas:
# - Imprime por generación: best, mean fitness y distancia.
# - Guarda logs CSV ("log_generations.csv") con best/mean por gen y el ADN del mejor de la última gen
#   en "best_dna.json" (para reproducirlo en Unity).

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
import json
import csv
import random
import numpy as np
from typing import Tuple, List

# -------------------------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def rand_in(rng: random.Random, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()

def gauss(rng: random.Random) -> float:
    # Box–Muller
    u1 = 1.0 - rng.random()
    u2 = 1.0 - rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

# -------------------------------------------------------------------------------------
# ADN / RANGOS (coincidirán con Unity luego)
# -------------------------------------------------------------------------------------

R_ALTURA = (1.30, 2.10)     # m
R_MASA   = (45.0, 95.0)     # kg
R_FRICP  = (0.6, 1.2)       # coef fricción pie
R_FREQ   = (0.6, 3.0)       # Hz

R_AMP    = (math.radians(5.0), math.radians(45.0))     # rad
R_OFF    = (math.radians(-10.0), math.radians(10.0))   # rad
R_FASE   = (0.0, 2.0 * math.pi)                        # rad

R_TAU    = (80.0, 220.0)    # N·m (límite de actuador por articulación)
R_KLIG   = (1_000.0, 6_000.0)  # N/m (rigidez ligamentaria aproximada)
R_CLIG   = (5.0, 60.0)      # N·s/m (amortiguamiento)

# Esqueleto: cadera, rodilla, tobillo por pierna (2 piernas => 6 articulaciones)
JOINTS = 6

@dataclass
class DNA:
    altura: float
    masa: float
    fric_pie: float
    freq: float
    amp: np.ndarray       # shape (6,)
    fase: np.ndarray      # shape (6,)
    offset: np.ndarray    # shape (6,)
    tau_max: np.ndarray   # shape (6,)
    k_lig: np.ndarray     # shape (6,)
    c_lig: np.ndarray     # shape (6,)

    @staticmethod
    def random(rng: random.Random) -> "DNA":
        def arr(lo, hi): return np.array([rand_in(rng, lo, hi) for _ in range(JOINTS)], dtype=np.float64)
        dna = DNA(
            altura = rand_in(rng, *R_ALTURA),
            masa   = rand_in(rng, *R_MASA),
            fric_pie = rand_in(rng, *R_FRICP),
            freq   = rand_in(rng, *R_FREQ),
            amp    = arr(*R_AMP),
            fase   = arr(*R_FASE),
            offset = arr(*R_OFF),
            tau_max= arr(*R_TAU),
            k_lig  = arr(*R_KLIG),
            c_lig  = arr(*R_CLIG),
        )
        return dna.clamped()

    def clamped(self) -> "DNA":
        self.altura  = clamp(self.altura, *R_ALTURA)
        self.masa    = clamp(self.masa, *R_MASA)
        self.fric_pie= clamp(self.fric_pie, *R_FRICP)
        self.freq    = clamp(self.freq, *R_FREQ)
        self.amp     = np.clip(self.amp, *R_AMP)
        self.fase    = np.clip(self.fase, *R_FASE)
        self.offset  = np.clip(self.offset, *R_OFF)
        self.tau_max = np.clip(self.tau_max, *R_TAU)
        self.k_lig   = np.clip(self.k_lig, *R_KLIG)
        self.c_lig   = np.clip(self.c_lig, *R_CLIG)
        return self

    def clone(self) -> "DNA":
        return DNA(
            altura=self.altura,
            masa=self.masa,
            fric_pie=self.fric_pie,
            freq=self.freq,
            amp=self.amp.copy(),
            fase=self.fase.copy(),
            offset=self.offset.copy(),
            tau_max=self.tau_max.copy(),
            k_lig=self.k_lig.copy(),
            c_lig=self.c_lig.copy(),
        )

    def mutate(self, rng: random.Random, p: float = 0.15, sigma_rel: float = 0.05):
        def mut_scalar(val, r):
            if rng.random() < p:
                span = r[1] - r[0]
                val += gauss(rng) * sigma_rel * span
            return clamp(val, *r)

        def mut_array(arr, r):
            lo, hi = r
            span = hi - lo
            mask = rng.random() < p
            # mut each independently
            a = arr.copy()
            for i in range(len(a)):
                if rng.random() < p:
                    a[i] += gauss(rng) * sigma_rel * span
            return np.clip(a, lo, hi)

        self.altura   = mut_scalar(self.altura, R_ALTURA)
        self.masa     = mut_scalar(self.masa,   R_MASA)
        self.fric_pie = mut_scalar(self.fric_pie, R_FRICP)
        self.freq     = mut_scalar(self.freq,   R_FREQ)

        self.amp      = mut_array(self.amp,   R_AMP)
        self.fase     = mut_array(self.fase,  R_FASE)
        self.offset   = mut_array(self.offset,R_OFF)
        self.tau_max  = mut_array(self.tau_max,R_TAU)
        self.k_lig    = mut_array(self.k_lig, R_KLIG)
        self.c_lig    = mut_array(self.c_lig, R_CLIG)

    @staticmethod
    def crossover_uniform(a: "DNA", b: "DNA", rng: random.Random) -> "DNA":
        c = a.clone()
        c.altura   = a.altura if rng.random() < 0.5 else b.altura
        c.masa     = a.masa   if rng.random() < 0.5 else b.masa
        c.fric_pie = a.fric_pie if rng.random() < 0.5 else b.fric_pie
        c.freq     = 0.5 * (a.freq + b.freq)  # suaviza

        sel = rng.random
        for i in range(JOINTS):
            c.amp[i]    = a.amp[i]    if sel() < 0.5 else b.amp[i]
            c.fase[i]   = a.fase[i]   if sel() < 0.5 else b.fase[i]
            c.offset[i] = a.offset[i] if sel() < 0.5 else b.offset[i]
            # Ser conservador con límites de torque (elige el menor)
            c.tau_max[i]= min(a.tau_max[i], b.tau_max[i])
            c.k_lig[i]  = a.k_lig[i]  if sel() < 0.5 else b.k_lig[i]
            c.c_lig[i]  = a.c_lig[i]  if sel() < 0.5 else b.c_lig[i]

        return c.clamped()

# -------------------------------------------------------------------------------------
# Simulador proxy 2D (plano sagital)
# Nota: no es un motor físico rígido, pero:
#  - Usa CPG para ángulos de cadera/rodilla/tobillo L/R
#  - Aproxima paso, avance y estabilidad con un SLIP simplificado
#  - Respeta límites de torque, fricción, energía, ligamentos (como springs + damping)
#  - Marca caída por CoM bajo o inclinación excesiva
# -------------------------------------------------------------------------------------

GRAV = 9.81

@dataclass
class SimConfig:
    dt: float = 1.0/240.0
    t_max: float = 12.0
    terreno_mu: float = 0.9        # fricción del suelo (se combina con fric_pie)
    max_body_pitch: float = math.radians(25.0)  # caída por inclinación
    com_min_ratio: float = 0.5     # CoM mínimo como fracción de altura
    z_lateral_drift_weight: float = 0.2  # no hay eje Z en 2D; mantenemos 0
    seed: int = 1234

class CreatureSim2D:
    """
    Modelo simplificado:
    - Segmentos: tronco, muslo, pierna, pie. Longitudes derivadas de altura.
    - CPG define ángulos objetivo para (cadera, rodilla, tobillo) en cada pierna.
    - Avance ~ función del ángulo de cadera de la pierna en fase de oscilación y longitud efectiva de pierna.
    - Contacto: si pie "entra" en el suelo, se considera apoyo; slip se estima si v_x del pie excede fricción.
    - Torque aplicado = clamp( Kp*e + Kd*edot, tau_max ), energía += |tau * qdot| dt.
    - “Ligamentos”: stiffness/damping adicionales que limitan excursiones (penalizan energía si excede).
    """
    def __init__(self, dna: DNA, cfg: SimConfig):
        self.dna = dna
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        # Longitudes morfológicas (proporciones humanas aprox.)
        H = dna.altura
        self.long_femur = 0.245 * H
        self.long_tibia = 0.246 * H
        self.long_pie   = 0.152 * H
        self.long_tronco= 0.30  * H

        # Estado
        self.t = 0.0
        self.x = 0.0            # avance
        self.pitch = 0.0        # inclinación del tronco
        self.com_y = self.com_height(0.0, 0.0)  # altura del centro de masa inicial
        self.energy = 0.0
        self.slip = 0.0
        self.fell = False

        # Fases (pierna izquierda y derecha en oposición)
        self.phase_L = 0.0
        self.phase_R = math.pi

        # Asignación de índices de articulación:
        # 0: hip_L, 1: knee_L, 2: ankle_L, 3: hip_R, 4: knee_R, 5: ankle_R
        self.q = np.zeros(6)        # ángulos actuales
        self.qdot = np.zeros(6)     # velocidades
        self.qref = np.zeros(6)     # objetivos

        # Control PD base (estos no son genes para simplificar)
        self.Kp = dna.k_lig.copy()      # usa k_lig como rigidez de joint
        self.Kd = dna.c_lig.copy() * 0.05  # amortiguamiento derivado

        # Fricción efectiva
        self.mu_eff = min(dna.fric_pie, self.cfg.terreno_mu)

        # Masa segmentada (aprox)
        m = dna.masa
        self.mass_tronco = 0.6 * m
        self.mass_legs   = 0.4 * m

    def com_height(self, hip_angle_L: float, hip_angle_R: float) -> float:
        # Altura CoM aproximada según extensión de piernas y tronco
        leg_len_eff = 0.5 * (self.long_femur + self.long_tibia) * (
            math.cos(abs(hip_angle_L)) + math.cos(abs(hip_angle_R))
        )
        return max(0.1, 0.5 * leg_len_eff + 0.5 * self.long_tronco)

    def step(self) -> None:
        dt = self.cfg.dt
        self.t += dt
        # Actualiza referencias CPG
        f = self.dna.freq
        # hip: sinusoide principal, rodilla/ankle en fase derivada
        # L
        self.qref[0] = self.dna.offset[0] + self.dna.amp[0] * math.sin(2*math.pi*f*self.t + self.dna.fase[0])
        self.qref[1] = self.dna.offset[1] + self.dna.amp[1] * math.sin(2*math.pi*f*self.t + self.dna.fase[1])
        self.qref[2] = self.dna.offset[2] + self.dna.amp[2] * math.sin(2*math.pi*f*self.t + self.dna.fase[2])
        # R
        self.qref[3] = self.dna.offset[3] + self.dna.amp[3] * math.sin(2*math.pi*f*self.t + self.dna.fase[3])
        self.qref[4] = self.dna.offset[4] + self.dna.amp[4] * math.sin(2*math.pi*f*self.t + self.dna.fase[4])
        self.qref[5] = self.dna.offset[5] + self.dna.amp[5] * math.sin(2*math.pi*f*self.t + self.dna.fase[5])

        # Torque PD + ligamentos (rigidez/amortiguamiento extra ya en Kp/Kd)
        tau = self.Kp * (self.qref - self.q) - self.Kd * self.qdot

        # Satura por límites de actuador y escálalo por masa^(2/3) (sección muscular ~ realismo)
        scale_strength = (self.dna.masa ** (2.0/3.0)) / (70.0 ** (2.0/3.0))
        tau_lim = self.dna.tau_max * scale_strength
        tau = np.clip(tau, -tau_lim, tau_lim)

        # “Dinámica” angular (muy simplificada)
        # qddot ~ tau / I  con I efectivo proporcional a masa y longitudes (constante simplificada)
        I_eff = 0.08 * self.dna.masa * (self.long_femur + self.long_tibia)  # momento inercial efectivo
        qddot = tau / (I_eff + 1e-6)
        self.qdot += qddot * dt
        self.q += self.qdot * dt

        # Energía mecánica consumida ~ ∑ |tau * qdot|
        self.energy += float(np.sum(np.abs(tau * self.qdot))) * dt

        # Estimar avance: cuando una cadera entra en extensión positiva y la otra en flexión, generamos paso
        hip_L = self.q[0]
        hip_R = self.q[3]
        # Longitud de paso ~ Δángulo de caderas * longitud de pierna efectiva (capado)
        step_len = clamp(abs(hip_L - hip_R) * (self.long_femur + self.long_tibia) * 0.25, 0.0, 1.25 * self.long_pie)
        # Velocidad hacia adelante ~ step_len * freq (capado por fricción/pérdidas)
        vx_raw = step_len * self.dna.freq
        # Slip si fuerza tangencial > mu * N; aprox: si vx_raw muy alto respecto a mu => parte se pierde en slip
        slip_factor = max(0.0, (vx_raw - self.mu_eff * 1.2) )
        self.slip += slip_factor * dt
        vx = vx_raw - 0.5 * slip_factor  # pierde parte por slip
        self.x += max(0.0, vx) * dt

        # Estimar pitch (inclinación) como desbalance entre caderas
        self.pitch = clamp(0.8 * (hip_L + hip_R), -math.radians(60), math.radians(60))

        # Actualiza CoM
        self.com_y = self.com_height(hip_L, hip_R)

        # Condiciones de caída
        if (self.com_y < self.cfg.com_min_ratio * self.dna.altura) or (abs(self.pitch) > self.cfg.max_body_pitch):
            self.fell = True

    def run_episode(self) -> Tuple[float, float, bool, float]:
        while self.t < self.cfg.t_max and not self.fell:
            self.step()
        return self.x, self.energy, self.fell, self.slip

# -------------------------------------------------------------------------------------
# Fitness (igual que mostraremos en Unity)
#   fitness = Dx - α * (E / (kg·m)) - β *σ_lat - γ*caída - δ*slip
# En 2D no hay desvío lateral => σ_lat = 0. Mantén la forma para compatibilidad.
# -------------------------------------------------------------------------------------

def compute_fitness(distance_x: float, energy: float, mass: float, fell: bool, slip: float,
                    alpha=0.3, beta=0.2, gamma=5.0, delta=0.1) -> float:
    Dx = max(0.0, distance_x)
    denom = mass * max(Dx, 0.5)
    e_per_kgm = energy / denom
    sigma_lat = 0.0
    f = Dx - alpha * e_per_kgm - beta * sigma_lat - delta * slip - (gamma if fell else 0.0)
    return float(f)

# -------------------------------------------------------------------------------------
# Algoritmo Genético
# -------------------------------------------------------------------------------------

@dataclass
class GAConfig:
    population: int = 60
    generations: int = 40
    elitism: int = 3
    tournament_k: int = 3
    mutation_p: float = 0.15
    mutation_sigma_rel: float = 0.05
    seed: int = 1234

class GeneticAlgorithm:
    def __init__(self, ga_cfg: GAConfig, sim_cfg: SimConfig):
        self.ga_cfg = ga_cfg
        self.sim_cfg = sim_cfg
        self.rng = random.Random(ga_cfg.seed)
        self.pop: List[DNA] = [DNA.random(self.rng) for _ in range(ga_cfg.population)]

    def evaluate(self, dna: DNA) -> Tuple[float, dict]:
        sim = CreatureSim2D(dna, self.sim_cfg)
        Dx, E, fell, slip = sim.run_episode()
        fit = compute_fitness(Dx, E, dna.masa, fell, slip)
        return fit, {
            "Dx": Dx,
            "E": E,
            "fell": fell,
            "slip": slip,
            "e_per_kgm": E / (dna.masa * max(Dx, 0.5))
        }

    def tournament(self, scored, k: int) -> DNA:
        best = None
        best_f = -1e18
        for _ in range(k):
            cand = self.rng.choice(scored)
            if cand[1] > best_f:
                best = cand[0]
                best_f = cand[1]
        return best.clone()

    def evolve(self):
        # Logs
        with open("log_generations.csv", "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["generation", "best_fitness", "mean_fitness", "best_Dx", "best_e_per_kgm", "best_fell", "best_slip"])

            for g in range(self.ga_cfg.generations):
                scored = []
                metrics = []
                for ind in self.pop:
                    fit, m = self.evaluate(ind)
                    scored.append((ind, fit))
                    metrics.append(m)

                scored.sort(key=lambda x: x[1], reverse=True)
                best_ind, best_fit = scored[0]
                mean_fit = sum(s for _, s in scored) / len(scored)

                best_m = metrics[np.argmax([m["Dx"] for m in metrics])]
                print(f"[Gen {g:02d}] Best={best_fit:.3f}  Mean={mean_fit:.3f}  Dx={best_m['Dx']:.2f}  E/kgm={best_m['e_per_kgm']:.3f}  Fell={best_m['fell']}  Slip={best_m['slip']:.2f}")
                writer.writerow([g, f"{best_fit:.6f}", f"{mean_fit:.6f}", f"{best_m['Dx']:.6f}", f"{best_m['e_per_kgm']:.6f}", best_m["fell"], f"{best_m['slip']:.6f}"])

                # Nueva población
                next_pop: List[DNA] = []
                # Elitismo
                for e in range(self.ga_cfg.elitism):
                    next_pop.append(scored[e][0].clone())

                # Rellenar con torneo + crossover + mutación
                while len(next_pop) < self.ga_cfg.population:
                    p1 = self.tournament(scored, self.ga_cfg.tournament_k)
                    p2 = self.tournament(scored, self.ga_cfg.tournament_k)
                    child = DNA.crossover_uniform(p1, p2, self.rng)
                    child.mutate(self.rng, self.ga_cfg.mutation_p, self.ga_cfg.mutation_sigma_rel)
                    next_pop.append(child.clamped())

                self.pop = next_pop

            # Guardar mejor ADN final para reproducir en Unity
            final_scored = [(ind, self.evaluate(ind)[0]) for ind in self.pop]
            final_scored.sort(key=lambda x: x[1], reverse=True)
            best_final = final_scored[0][0]
            with open("best_dna.json", "w", encoding="utf-8") as fj:
                json.dump({
                    "altura": best_final.altura,
                    "masa": best_final.masa,
                    "fric_pie": best_final.fric_pie,
                    "freq": best_final.freq,
                    "amp": best_final.amp.tolist(),
                    "fase": best_final.fase.tolist(),
                    "offset": best_final.offset.tolist(),
                    "tau_max": best_final.tau_max.tolist(),
                    "k_lig": best_final.k_lig.tolist(),
                    "c_lig": best_final.c_lig.tolist()
                }, fj, indent=2, ensure_ascii=False)
            print("→ Guardados: log_generations.csv y best_dna.json")

# -------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    ga_cfg = GAConfig(
        population=60,
        generations=30,    # puedes subir a 50+
        elitism=3,
        tournament_k=3,
        mutation_p=0.15,
        mutation_sigma_rel=0.05,
        seed=1234
    )
    sim_cfg = SimConfig(
        dt=1.0/240.0,
        t_max=12.0,
        terreno_mu=0.9,
        max_body_pitch=math.radians(25.0),
        com_min_ratio=0.5,
        seed=42
    )
    ga = GeneticAlgorithm(ga_cfg, sim_cfg)
    ga.evolve()
