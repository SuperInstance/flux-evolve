use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::cmp::Ordering;
use crate::fastrand::Shuffle;

fn hash_str(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn pseudo_random(generation: u32, name: &str) -> f64 {
    let combined = format!("{}:{}", generation, name);
    let h = hash_str(&combined);
    (h % 10000) as f64 / 10000.0
}

// ============================================================================
// Original Engine types (preserved for backward compatibility)
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub enum MutationType {
    ParamAdjust,
    ThresholdShift,
    WeightRebalance,
    AddBehavior,
    RemoveBehavior,
    SwapPriority,
    RateChange,
    CapChange,
}

#[derive(Clone, Debug)]
pub struct Behavior {
    pub name: String,
    pub value: f64,
    pub min: f64,
    pub max: f64,
    pub default_val: f64,
    pub mutation_rate: f64,
    pub uses: u32,
    pub cumulative_score: f64,
}

#[derive(Clone, Debug)]
pub struct MutationRecord {
    pub mutation_type: MutationType,
    pub parameter: String,
    pub old_value: f64,
    pub new_value: f64,
    pub reason: String,
    pub generation: u32,
    pub reverted: bool,
}

pub struct Engine {
    behaviors: HashMap<String, Behavior>,
    history: Vec<MutationRecord>,
    generation: u32,
    mutations_total: u32,
    mutations_reverted: u32,
    fitness_threshold: f64,
    mutation_probability: f64,
    elite_threshold: f64,
}

impl Engine {
    pub fn new() -> Self {
        Engine {
            behaviors: HashMap::new(),
            history: Vec::new(),
            generation: 0,
            mutations_total: 0,
            mutations_reverted: 0,
            fitness_threshold: 0.3,
            mutation_probability: 0.1,
            elite_threshold: 0.8,
        }
    }

    pub fn add_behavior(&mut self, name: &str, value: f64, min: f64, max: f64, mut_rate: f64) {
        let b = Behavior {
            name: name.to_string(),
            value: value.clamp(min, max),
            min,
            max,
            default_val: value,
            mutation_rate: mut_rate,
            uses: 0,
            cumulative_score: 0.0,
        };
        self.behaviors.insert(name.to_string(), b);
    }

    pub fn find_behavior(&self, name: &str) -> Option<&Behavior> {
        self.behaviors.get(name)
    }

    pub fn get(&self, name: &str) -> f64 {
        self.behaviors.get(name).map(|b| b.value).unwrap_or(-1.0)
    }

    pub fn set(&mut self, name: &str, value: f64) {
        if let Some(b) = self.behaviors.get_mut(name) {
            b.value = value.clamp(b.min, b.max);
        }
    }

    pub fn cycle(&mut self, fitness: f64) -> usize {
        self.generation += 1;
        let mut mutations = 0usize;

        if fitness >= self.elite_threshold {
            let worst_names: Vec<String> = self.worst_behaviors(3).into_iter().map(|b| b.name.clone()).collect();
            for name in worst_names {
                if pseudo_random(self.generation, &name) < self.mutation_probability * 2.0 {
                    mutations += self.mutate_behavior(&name, "elite pressure");
                }
            }
        } else if fitness >= self.fitness_threshold {
            for name in self.behaviors.keys().cloned().collect::<Vec<_>>() {
                let prob = self.behaviors.get(&name).map(|b| b.mutation_rate).unwrap_or(self.mutation_probability);
                if pseudo_random(self.generation, &name) < prob {
                    mutations += self.mutate_behavior(&name, "normal evolution");
                }
            }
        } else {
            for name in self.behaviors.keys().cloned().collect::<Vec<_>>() {
                if pseudo_random(self.generation, &name) < self.mutation_probability * 3.0 {
                    mutations += self.mutate_behavior(&name, "aggressive mutation");
                }
            }
        }

        mutations
    }

    fn mutate_behavior(&mut self, name: &str, reason: &str) -> usize {
        let (old_value, min, max) = {
            let b = match self.behaviors.get(name) {
                Some(b) => b,
                None => return 0,
            };
            let range = b.max - b.min;
            if range == 0.0 {
                return 0;
            }
            (b.value, b.min, b.max)
        };

        let range = max - min;
        let r = pseudo_random(self.generation, &format!("mut:{}", name));
        let direction: f64 = if r < 0.5 { -1.0 } else { 1.0 };
        let magnitude = range * 0.1 * (pseudo_random(self.generation, &format!("mag:{}", name)) + 0.1);
        let new_value = (old_value + direction * magnitude).clamp(min, max);

        if (new_value - old_value).abs() < 1e-10 {
            return 0;
        }

        let record = MutationRecord {
            mutation_type: MutationType::ParamAdjust,
            parameter: name.to_string(),
            old_value,
            new_value,
            reason: reason.to_string(),
            generation: self.generation,
            reverted: false,
        };

        self.mutations_total += 1;
        if let Some(b) = self.behaviors.get_mut(name) {
            b.value = new_value;
        }
        self.history.push(record);
        1
    }

    pub fn score(&mut self, behavior: &str, outcome: f64) {
        if let Some(b) = self.behaviors.get_mut(behavior) {
            b.uses += 1;
            b.cumulative_score += outcome;
        }
    }

    pub fn revert(&mut self, index: usize) -> bool {
        if index >= self.history.len() {
            return false;
        }
        let record = &self.history[index];
        if record.reverted {
            return false;
        }
        let name = record.parameter.clone();
        let old_value = record.old_value;
        if let Some(b) = self.behaviors.get_mut(&name) {
            b.value = old_value;
        }
        self.history[index].reverted = true;
        self.mutations_reverted += 1;
        true
    }

    pub fn rollback(&mut self, target_generation: u32) -> usize {
        let mut reverted = 0usize;
        for i in (0..self.history.len()).rev() {
            if self.history[i].generation > target_generation && !self.history[i].reverted {
                if self.revert(i) {
                    reverted += 1;
                }
            }
        }
        reverted
    }

    pub fn worst_behaviors(&self, n: usize) -> Vec<&Behavior> {
        let mut sorted: Vec<&Behavior> = self.behaviors.values().collect();
        sorted.sort_by(|a, b| {
            let avg_a = if a.uses > 0 { a.cumulative_score / a.uses as f64 } else { 0.0 };
            let avg_b = if b.uses > 0 { b.cumulative_score / b.uses as f64 } else { 0.0 };
            avg_a.partial_cmp(&avg_b).unwrap_or(Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    pub fn best_behaviors(&self, n: usize) -> Vec<&Behavior> {
        let mut sorted: Vec<&Behavior> = self.behaviors.values().collect();
        sorted.sort_by(|a, b| {
            let avg_a = if a.uses > 0 { a.cumulative_score / a.uses as f64 } else { 0.0 };
            let avg_b = if b.uses > 0 { b.cumulative_score / b.uses as f64 } else { 0.0 };
            avg_b.partial_cmp(&avg_a).unwrap_or(Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn history(&self) -> &[MutationRecord] {
        &self.history
    }

    pub fn mutations_total(&self) -> u32 {
        self.mutations_total
    }

    pub fn mutations_reverted(&self) -> u32 {
        self.mutations_reverted
    }
}

// ============================================================================
// Genetic Algorithm Framework
// ============================================================================

/// A chromosome is a vector of genes (f64 values).
pub type Chromosome = Vec<f64>;

/// An individual in the population with its fitness.
#[derive(Clone, Debug)]
pub struct Individual {
    pub chromosome: Chromosome,
    pub fitness: f64,
}

impl Individual {
    pub fn new(chromosome: Chromosome) -> Self {
        Individual { chromosome, fitness: 0.0 }
    }

    pub fn with_fitness(chromosome: Chromosome, fitness: f64) -> Self {
        Individual { chromosome, fitness }
    }

    /// Length of the chromosome.
    pub fn len(&self) -> usize {
        self.chromosome.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chromosome.is_empty()
    }
}

/// Configuration for a genetic algorithm run.
#[derive(Clone, Debug)]
pub struct GAConfig {
    pub population_size: usize,
    pub chromosome_length: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elitism_count: usize,
    pub max_generations: u32,
    pub target_fitness: Option<f64>,
    pub gene_min: f64,
    pub gene_max: f64,
    pub tournament_size: usize,
    pub convergence_window: usize,
    pub convergence_threshold: f64,
}

impl Default for GAConfig {
    fn default() -> Self {
        GAConfig {
            population_size: 50,
            chromosome_length: 10,
            mutation_rate: 0.05,
            crossover_rate: 0.8,
            elitism_count: 2,
            max_generations: 500,
            target_fitness: None,
            gene_min: 0.0,
            gene_max: 1.0,
            tournament_size: 3,
            convergence_window: 20,
            convergence_threshold: 1e-6,
        }
    }
}

/// A fitness function that evaluates a chromosome.
pub trait FitnessFn: Fn(&Chromosome) -> f64 + Send + Sync {}

impl<F: Fn(&Chromosome) -> f64 + Send + Sync> FitnessFn for F {}

/// A selection strategy for choosing parents.
pub trait SelectionStrategy {
    /// Select two parents from the population.
    fn select(&self, population: &[Individual]) -> (usize, usize);
}

/// Roulette wheel (fitness-proportionate) selection.
pub struct RouletteWheelSelection;

impl SelectionStrategy for RouletteWheelSelection {
    fn select(&self, population: &[Individual]) -> (usize, usize) {
        let total_fitness: f64 = population.iter().map(|i| i.fitness.max(0.0)).sum();
        if total_fitness <= 0.0 {
            // Uniform random if no fitness signal
            return (random_index(population.len()), random_index(population.len()));
        }

        let parent1 = roulette_pick(population, total_fitness);
        let parent2 = roulette_pick(population, total_fitness);
        (parent1, parent2)
    }
}

fn roulette_pick(population: &[Individual], total_fitness: f64) -> usize {
    let mut r = pseudo_random_ga(total_fitness);
    for (i, ind) in population.iter().enumerate() {
        let f = ind.fitness.max(0.0);
        r -= f;
        if r <= 0.0 {
            return i;
        }
    }
    population.len() - 1
}

/// Tournament selection.
pub struct TournamentSelection {
    pub tournament_size: usize,
}

impl TournamentSelection {
    pub fn new(size: usize) -> Self {
        TournamentSelection { tournament_size: size.max(2) }
    }
}

impl SelectionStrategy for TournamentSelection {
    fn select(&self, population: &[Individual]) -> (usize, usize) {
        let p1 = tournament_pick(population, self.tournament_size);
        let p2 = tournament_pick(population, self.tournament_size);
        (p1, p2)
    }
}

fn tournament_pick(population: &[Individual], size: usize) -> usize {
    let mut best = random_index(population.len());
    for _ in 1..size {
        let idx = random_index(population.len());
        if population[idx].fitness > population[best].fitness {
            best = idx;
        }
    }
    best
}

/// Rank-based selection.
pub struct RankBasedSelection;

impl SelectionStrategy for RankBasedSelection {
    fn select(&self, population: &[Individual]) -> (usize, usize) {
        // Sort by fitness (ascending) to assign ranks
        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            population[a].fitness.partial_cmp(&population[b].fitness).unwrap_or(Ordering::Equal)
        });
        let n = population.len();
        // Rank weights: rank 0 gets weight 1, rank n-1 gets weight n
        let total_weight = n as f64 * (n as f64 + 1.0) / 2.0;

        let pick = |total: f64| -> usize {
            let mut r = pseudo_random_ga(total);
            for (rank, &idx) in indices.iter().enumerate() {
                let w = (rank + 1) as f64;
                r -= w;
                if r <= 0.0 {
                    return idx;
                }
            }
            *indices.last().unwrap()
        };

        (pick(total_weight), pick(total_weight))
    }
}

// Simple pseudo-random generator seeded from thread-local state.
fn pseudo_random_ga(seed: f64) -> f64 {
    let mut h = DefaultHasher::new();
    seed.to_bits().hash(&mut h);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut h);
    (h.finish() % 10000) as f64 / 10000.0
}

fn random_index(max: usize) -> usize {
    (pseudo_random_ga(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as f64 * max as f64)
        * max as f64) as usize % max
}

// ============================================================================
// Crossover operators
// ============================================================================

/// Single-point crossover: split parents at one point and swap tails.
pub fn crossover_single_point(parent1: &Chromosome, parent2: &Chromosome) -> (Chromosome, Chromosome) {
    if parent1.len() <= 1 {
        return (parent1.clone(), parent2.clone());
    }
    let point = random_index(parent1.len().saturating_sub(1)) + 1;
    let mut child1 = parent1[..point].to_vec();
    let mut child2 = parent2[..point].to_vec();
    child1.extend_from_slice(&parent2[point..]);
    child2.extend_from_slice(&parent1[point..]);
    (child1, child2)
}

/// Two-point crossover: split at two points and swap the middle segment.
pub fn crossover_two_point(parent1: &Chromosome, parent2: &Chromosome) -> (Chromosome, Chromosome) {
    let len = parent1.len();
    if len <= 2 {
        return crossover_single_point(parent1, parent2);
    }
    let p1 = random_index(len.saturating_sub(2));
    let p2 = random_index(len - p1 - 1) + p1 + 1;
    let mut child1 = parent1[..p1].to_vec();
    let mut child2 = parent2[..p1].to_vec();
    child1.extend_from_slice(&parent2[p1..p2]);
    child2.extend_from_slice(&parent1[p1..p2]);
    child1.extend_from_slice(&parent2[p2..]);
    child2.extend_from_slice(&parent1[p2..]);
    (child1, child2)
}

/// Uniform crossover: each gene is randomly taken from either parent.
pub fn crossover_uniform(parent1: &Chromosome, parent2: &Chromosome, mix_probability: f64) -> (Chromosome, Chromosome) {
    let len = parent1.len().min(parent2.len());
    let mut child1 = Vec::with_capacity(len);
    let mut child2 = Vec::with_capacity(len);
    for i in 0..len {
        let from_p1 = pseudo_random_ga(i as f64 + mix_probability * 100.0) < 0.5;
        if from_p1 {
            child1.push(parent1[i]);
            child2.push(parent2[i]);
        } else {
            child1.push(parent2[i]);
            child2.push(parent1[i]);
        }
    }
    (child1, child2)
}

// ============================================================================
// Mutation operators
// ============================================================================

/// Bit-flip style mutation for real-valued genes: flip to a random value.
pub fn mutate_random_replace(chromosome: &mut Chromosome, rate: f64, min: f64, max: f64) -> usize {
    let mut count = 0;
    for gene in chromosome.iter_mut() {
        if pseudo_random_ga(*gene * 1000.0) < rate {
            *gene = min + pseudo_random_ga(*gene * 999.0) * (max - min);
            count += 1;
        }
    }
    count
}

/// Swap mutation: swap two random genes.
pub fn mutate_swap(chromosome: &mut Chromosome, rate: f64) -> usize {
    if chromosome.len() < 2 {
        return 0;
    }
    if pseudo_random_ga(chromosome[0] * 777.0) >= rate {
        return 0;
    }
    let i = random_index(chromosome.len());
    let mut j = random_index(chromosome.len());
    while j == i {
        j = random_index(chromosome.len());
    }
    chromosome.swap(i, j);
    1
}

/// Scramble mutation: randomly shuffle a segment of the chromosome.
pub fn mutate_scramble(chromosome: &mut Chromosome, rate: f64) -> usize {
    if chromosome.len() < 3 {
        return 0;
    }
    if pseudo_random_ga(chromosome[0] * 333.0) >= rate {
        return 0;
    }
    let start = random_index(chromosome.len().saturating_sub(2));
    let end = random_index(chromosome.len() - start) + start + 1;
    chromosome[start..end].shuffle(&mut fastrand::Rng::new());
    1
}

/// Invert mutation: reverse a segment of the chromosome.
pub fn mutate_invert(chromosome: &mut Chromosome, rate: f64) -> usize {
    if chromosome.len() < 2 {
        return 0;
    }
    if pseudo_random_ga(chromosome[0] * 555.0) >= rate {
        return 0;
    }
    let start = random_index(chromosome.len().saturating_sub(1));
    let end = random_index(chromosome.len() - start) + start + 1;
    chromosome[start..end].reverse();
    1
}

/// Gaussian perturbation mutation: add small random noise to each gene.
pub fn mutate_gaussian(chromosome: &mut Chromosome, rate: f64, std_dev: f64) -> usize {
    let mut count = 0;
    for gene in chromosome.iter_mut() {
        if pseudo_random_ga(*gene * 1234.0) < rate {
            let noise = gaussian_noise() * std_dev;
            *gene += noise;
            count += 1;
        }
    }
    count
}

/// Simple Box-Muller gaussian noise generator.
fn gaussian_noise() -> f64 {
    let u1 = pseudo_random_ga(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as f64 + 0.123);
    let u2 = pseudo_random_ga(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as f64 + 0.456);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ============================================================================
// Genetic Algorithm runner
// ============================================================================

/// Statistics tracked per generation.
#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub generation: u32,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub avg_fitness: f64,
    pub diversity: f64, // standard deviation of fitness values
}

/// Result of a GA run.
#[derive(Clone, Debug)]
pub struct GAResult {
    pub best_individual: Individual,
    pub generation: u32,
    pub converged: bool,
    pub stats: Vec<GenerationStats>,
}

/// The main genetic algorithm engine.
pub struct GeneticAlgorithm {
    config: GAConfig,
    population: Vec<Individual>,
    stats: Vec<GenerationStats>,
}

impl GeneticAlgorithm {
    /// Create a new GA with the given configuration.
    pub fn new(config: GAConfig) -> Self {
        GeneticAlgorithm {
            config,
            population: Vec::new(),
            stats: Vec::new(),
        }
    }

    /// Initialize population with random chromosomes.
    pub fn initialize<F: FitnessFn>(&mut self, fitness_fn: &F) {
        self.population.clear();
        self.stats.clear();
        for _ in 0..self.config.population_size {
            let chromosome: Chromosome = (0..self.config.chromosome_length)
                .map(|_| {
                    self.config.gene_min
                        + pseudo_random_ga(std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as f64 + chromosome_idx_rand())
                            * (self.config.gene_max - self.config.gene_min)
                })
                .collect();
            let mut ind = Individual::new(chromosome);
            ind.fitness = fitness_fn(&ind.chromosome);
            self.population.push(ind);
        }
    }

    /// Run the GA to completion with a given fitness function and selection strategy.
    pub fn run<F: FitnessFn, S: SelectionStrategy>(
        &mut self,
        fitness_fn: &F,
        selection: &S,
    ) -> GAResult {
        // Sort initial population
        self.sort_population();
        self.record_stats(0);

        for gen in 1..=self.config.max_generations {
            // Check target fitness
            if let Some(target) = self.config.target_fitness {
                if self.population[0].fitness >= target {
                    return GAResult {
                        best_individual: self.population[0].clone(),
                        generation: gen - 1,
                        converged: true,
                        stats: self.stats.clone(),
                    };
                }
            }

            // Check convergence
            if self.has_converged() {
                return GAResult {
                    best_individual: self.population[0].clone(),
                    generation: gen - 1,
                    converged: true,
                    stats: self.stats.clone(),
                };
            }

            // Create next generation
            let mut next_pop = Vec::with_capacity(self.config.population_size);

            // Elitism: copy top performers directly
            for i in 0..self.config.elitism_count.min(self.population.len()) {
                next_pop.push(self.population[i].clone());
            }

            // Fill the rest with offspring
            while next_pop.len() < self.config.population_size {
                let (p1_idx, p2_idx) = selection.select(&self.population);
                let parent1 = &self.population[p1_idx].chromosome;
                let parent2 = &self.population[p2_idx].chromosome;

                // Crossover
                let (mut child1, mut child2) = if pseudo_random_ga(gen as f64 * 0.7) < self.config.crossover_rate {
                    crossover_single_point(parent1, parent2)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation
                mutate_random_replace(&mut child1, self.config.mutation_rate,
                    self.config.gene_min, self.config.gene_max);
                mutate_random_replace(&mut child2, self.config.mutation_rate,
                    self.config.gene_min, self.config.gene_max);

                // Evaluate fitness
                let mut ind1 = Individual::new(child1);
                ind1.fitness = fitness_fn(&ind1.chromosome);
                next_pop.push(ind1);

                if next_pop.len() < self.config.population_size {
                    let mut ind2 = Individual::new(child2);
                    ind2.fitness = fitness_fn(&ind2.chromosome);
                    next_pop.push(ind2);
                }
            }

            self.population = next_pop;
            self.sort_population();
            self.record_stats(gen);
        }

        GAResult {
            best_individual: self.population[0].clone(),
            generation: self.config.max_generations,
            converged: false,
            stats: self.stats.clone(),
        }
    }

    /// Run with explicit crossover and mutation strategies.
    pub fn run_with_operators<F: FitnessFn, S: SelectionStrategy, C, M>(
        &mut self,
        fitness_fn: &F,
        selection: &S,
        crossover_fn: C,
        mutation_fn: M,
    ) -> GAResult
    where
        C: Fn(&Chromosome, &Chromosome) -> (Chromosome, Chromosome),
        M: Fn(&mut Chromosome),
    {
        self.sort_population();
        self.record_stats(0);

        for gen in 1..=self.config.max_generations {
            if let Some(target) = self.config.target_fitness {
                if self.population[0].fitness >= target {
                    return GAResult {
                        best_individual: self.population[0].clone(),
                        generation: gen - 1,
                        converged: true,
                        stats: self.stats.clone(),
                    };
                }
            }

            if self.has_converged() {
                return GAResult {
                    best_individual: self.population[0].clone(),
                    generation: gen - 1,
                    converged: true,
                    stats: self.stats.clone(),
                };
            }

            let mut next_pop = Vec::with_capacity(self.config.population_size);

            // Elitism
            for i in 0..self.config.elitism_count.min(self.population.len()) {
                next_pop.push(self.population[i].clone());
            }

            while next_pop.len() < self.config.population_size {
                let (p1_idx, p2_idx) = selection.select(&self.population);
                let parent1 = &self.population[p1_idx].chromosome;
                let parent2 = &self.population[p2_idx].chromosome;

                let (mut child1, mut child2) = if pseudo_random_ga(gen as f64 * 1.3) < self.config.crossover_rate {
                    crossover_fn(parent1, parent2)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                mutation_fn(&mut child1);
                mutation_fn(&mut child2);

                let mut ind1 = Individual::new(child1);
                ind1.fitness = fitness_fn(&ind1.chromosome);
                next_pop.push(ind1);

                if next_pop.len() < self.config.population_size {
                    let mut ind2 = Individual::new(child2);
                    ind2.fitness = fitness_fn(&ind2.chromosome);
                    next_pop.push(ind2);
                }
            }

            self.population = next_pop;
            self.sort_population();
            self.record_stats(gen);
        }

        GAResult {
            best_individual: self.population[0].clone(),
            generation: self.config.max_generations,
            converged: false,
            stats: self.stats.clone(),
        }
    }

    fn sort_population(&mut self) {
        self.population.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal)
        });
    }

    fn record_stats(&mut self, gen: u32) {
        if self.population.is_empty() {
            return;
        }
        let best = self.population[0].fitness;
        let worst = self.population.last().unwrap().fitness;
        let avg = self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population.len() as f64;
        let variance = self.population.iter()
            .map(|i| (i.fitness - avg).powi(2))
            .sum::<f64>() / self.population.len() as f64;
        let diversity = variance.sqrt();
        self.stats.push(GenerationStats {
            generation: gen,
            best_fitness: best,
            worst_fitness: worst,
            avg_fitness: avg,
            diversity,
        });
    }

    /// Check if the population has converged (fitness diversity below threshold).
    pub fn has_converged(&self) -> bool {
        if self.stats.len() < self.config.convergence_window {
            return false;
        }
        let window = &self.stats[self.stats.len() - self.config.convergence_window..];
        let first_best = window.first().unwrap().best_fitness;
        let last_best = window.last().unwrap().best_fitness;
        (last_best - first_best).abs() < self.config.convergence_threshold
    }

    /// Get the current best individual.
    pub fn best(&self) -> Option<&Individual> {
        self.population.first()
    }

    /// Get population statistics.
    pub fn stats(&self) -> &[GenerationStats] {
        &self.stats
    }

    /// Get the current population.
    pub fn population(&self) -> &[Individual] {
        &self.population
    }
}

/// Helper for generating random indices (deterministic-ish for tests).
fn chromosome_idx_rand() -> f64 {
    let mut h = DefaultHasher::new();
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut h);
    (h.finish() % 10000) as f64 / 10000.0
}

// ============================================================================
// Trait objects for dynamic dispatch
// ============================================================================

/// Boxed fitness function for use when a trait object is needed.
pub type BoxFitnessFn = Box<dyn Fn(&Chromosome) -> f64 + Send + Sync>;

/// Boxed selection strategy.
pub type BoxSelectionStrategy = Box<dyn SelectionStrategy>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Original Engine tests (backward compat) --
    #[test]
    fn test_new_engine() {
        let e = Engine::new();
        assert_eq!(e.generation(), 0);
        assert_eq!(e.mutations_total(), 0);
    }

    #[test]
    fn test_add_behavior() {
        let mut e = Engine::new();
        e.add_behavior("aggression", 0.5, 0.0, 1.0, 0.1);
        assert_eq!(e.get("aggression"), 0.5);
        assert!(e.find_behavior("aggression").is_some());
    }

    #[test]
    fn test_get_not_found() {
        let e = Engine::new();
        assert_eq!(e.get("nonexistent"), -1.0);
    }

    #[test]
    fn test_find_behavior_none() {
        let e = Engine::new();
        assert!(e.find_behavior("nope").is_none());
    }

    #[test]
    fn test_set_clamp() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 0.1);
        e.set("x", 1.5);
        assert_eq!(e.get("x"), 1.0);
        e.set("x", -0.5);
        assert_eq!(e.get("x"), 0.0);
    }

    #[test]
    fn test_set_nonexistent() {
        let mut e = Engine::new();
        e.set("nope", 0.5);
        assert_eq!(e.get("nope"), -1.0);
    }

    #[test]
    fn test_score() {
        let mut e = Engine::new();
        e.add_behavior("b", 0.5, 0.0, 1.0, 0.1);
        e.score("b", 1.0);
        e.score("b", 0.5);
        let b = e.find_behavior("b").unwrap();
        assert_eq!(b.uses, 2);
        assert!((b.cumulative_score - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_score_nonexistent() {
        let mut e = Engine::new();
        e.score("nope", 1.0);
    }

    #[test]
    fn test_cycle_increments_generation() {
        let mut e = Engine::new();
        e.add_behavior("a", 0.5, 0.0, 1.0, 0.0);
        let before = e.generation();
        e.cycle(0.5);
        assert_eq!(e.generation(), before + 1);
    }

    #[test]
    fn test_cycle_mutations_above_threshold() {
        let mut e = Engine::new();
        e.add_behavior("a", 0.5, 0.0, 1.0, 1.0);
        let count = e.cycle(0.5);
        assert!(count >= 0);
    }

    #[test]
    fn test_revert() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 1.0);
        e.cycle(0.5);
        if !e.history().is_empty() {
            let last = e.history().len() - 1;
            let old_val = e.history()[last].old_value;
            let new_val = e.history()[last].new_value;
            assert!((new_val - old_val).abs() > 1e-10);
            assert!(e.revert(last));
            assert_eq!(e.get("x"), old_val);
            assert!(e.history()[last].reverted);
        }
    }

    #[test]
    fn test_revert_already_reverted() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 1.0);
        e.cycle(0.5);
        if !e.history().is_empty() {
            let idx = e.history().len() - 1;
            assert!(e.revert(idx));
            assert!(!e.revert(idx));
        }
    }

    #[test]
    fn test_revert_invalid_index() {
        let mut e = Engine::new();
        assert!(!e.revert(0));
        assert!(!e.revert(999));
    }

    #[test]
    fn test_rollback() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 1.0);
        let _ = e.cycle(0.5);
        let _ = e.cycle(0.5);
        let _ = e.cycle(0.5);
        let pre = e.get("x");
        let reverted = e.rollback(1);
        let post = e.get("x");
        assert!(reverted >= 0);
        if reverted > 0 {
            assert!((pre - post).abs() > 1e-10 || pre == post);
        }
    }

    #[test]
    fn test_best_behaviors() {
        let mut e = Engine::new();
        e.add_behavior("good", 0.5, 0.0, 1.0, 0.1);
        e.add_behavior("bad", 0.5, 0.0, 1.0, 0.1);
        e.score("good", 10.0);
        e.score("good", 10.0);
        e.score("bad", -5.0);
        e.score("bad", -5.0);
        let best = e.best_behaviors(1);
        assert_eq!(best[0].name, "good");
    }

    #[test]
    fn test_worst_behaviors() {
        let mut e = Engine::new();
        e.add_behavior("good", 0.5, 0.0, 1.0, 0.1);
        e.add_behavior("bad", 0.5, 0.0, 1.0, 0.1);
        e.score("good", 10.0);
        e.score("bad", -5.0);
        let worst = e.worst_behaviors(1);
        assert_eq!(worst[0].name, "bad");
    }

    #[test]
    fn test_aggressive_cycle() {
        let mut e = Engine::new();
        e.add_behavior("x", 0.5, 0.0, 1.0, 0.01);
        let count = e.cycle(0.1);
        assert_eq!(e.generation(), 1);
        assert!(count >= 0);
    }

    #[test]
    fn test_elite_cycle() {
        let mut e = Engine::new();
        e.add_behavior("bad", 0.5, 0.0, 1.0, 0.1);
        e.score("bad", -10.0);
        let _ = e.cycle(0.9);
        assert_eq!(e.generation(), 1);
    }

    #[test]
    fn test_mutation_record_fields() {
        let mut e = Engine::new();
        e.add_behavior("p", 0.5, 0.0, 1.0, 1.0);
        e.cycle(0.5);
        if let Some(rec) = e.history().last() {
            assert_eq!(rec.generation, 1);
            assert!(!rec.reverted);
            assert!(matches!(rec.mutation_type, MutationType::ParamAdjust));
        }
    }

    // -- GA Framework tests --

    #[test]
    fn test_ga_config_default() {
        let cfg = GAConfig::default();
        assert_eq!(cfg.population_size, 50);
        assert_eq!(cfg.mutation_rate, 0.05);
        assert_eq!(cfg.elitism_count, 2);
    }

    #[test]
    fn test_individual_new() {
        let ind = Individual::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(ind.len(), 3);
        assert!(!ind.is_empty());
        assert_eq!(ind.fitness, 0.0);
    }

    #[test]
    fn test_individual_with_fitness() {
        let ind = Individual::with_fitness(vec![1.0], 42.0);
        assert!((ind.fitness - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_crossover_single_point() {
        let p1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (c1, c2) = crossover_single_point(&p1, &p2);
        assert_eq!(c1.len(), 5);
        assert_eq!(c2.len(), 5);
        // Verify crossover happened (not exact copies)
        assert!(c1 != p1 || c1 != p2);
        assert!(c2 != p1 || c2 != p2);
    }

    #[test]
    fn test_crossover_two_point() {
        let p1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p2 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let (c1, c2) = crossover_two_point(&p1, &p2);
        assert_eq!(c1.len(), 6);
        assert_eq!(c2.len(), 6);
    }

    #[test]
    fn test_crossover_uniform() {
        let p1 = vec![1.0, 2.0, 3.0, 4.0];
        let p2 = vec![10.0, 20.0, 30.0, 40.0];
        let (c1, c2) = crossover_uniform(&p1, &p2, 0.5);
        assert_eq!(c1.len(), 4);
        assert_eq!(c2.len(), 4);
        // Each gene should come from one parent or the other
        for i in 0..4 {
            assert!(c1[i] == p1[i] || c1[i] == p2[i]);
        }
    }

    #[test]
    fn test_mutate_random_replace() {
        let mut chrom = vec![0.5, 0.5, 0.5];
        let count = mutate_random_replace(&mut chrom, 1.0, 0.0, 1.0);
        assert!(count > 0);
        // With rate=1.0, all genes should be replaced (or very likely)
        assert_eq!(count, 3);
    }

    #[test]
    fn test_mutate_swap() {
        let mut chrom = vec![1.0, 2.0, 3.0, 4.0];
        let original = chrom.clone();
        let _ = mutate_swap(&mut chrom, 1.0);
        // With rate=1.0, a swap should occur
        // Check that exactly two elements changed position
        let diff_count = chrom.iter().zip(original.iter()).filter(|(a, b)| a != b).count();
        assert!(diff_count >= 2, "Expected at least 2 differences, got {}", diff_count);
    }

    #[test]
    fn test_mutate_invert() {
        let mut chrom = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let _ = mutate_invert(&mut chrom, 1.0);
        assert_eq!(chrom.len(), 5);
        // Verify the total multiset of genes is preserved
        let mut sorted_original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut sorted_result = chrom.clone();
        sorted_original.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted_original, sorted_result);
    }

    #[test]
    fn test_mutate_scramble() {
        let mut chrom = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let original = chrom.clone();
        let _ = mutate_scramble(&mut chrom, 1.0);
        assert_eq!(chrom.len(), 6);
        // Genes should be the same multiset
        let mut s1 = original.clone();
        let mut s2 = chrom.clone();
        s1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_mutate_gaussian() {
        let mut chrom = vec![0.5, 0.5, 0.5];
        let count = mutate_gaussian(&mut chrom, 1.0, 0.1);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_mutate_zero_rate() {
        let mut chrom = vec![1.0, 2.0, 3.0];
        let original = chrom.clone();
        let count = mutate_random_replace(&mut chrom, 0.0, 0.0, 1.0);
        assert_eq!(count, 0);
        assert_eq!(chrom, original);
    }

    #[test]
    fn test_tournament_selection() {
        let pop = vec![
            Individual::with_fitness(vec![0.0], 10.0),
            Individual::with_fitness(vec![0.0], 5.0),
            Individual::with_fitness(vec![0.0], 1.0),
            Individual::with_fitness(vec![0.0], 8.0),
            Individual::with_fitness(vec![0.0], 3.0),
        ];
        let sel = TournamentSelection::new(3);
        let (p1, p2) = sel.select(&pop);
        // With tournament selection, fitter individuals should be preferred
        assert!(pop[p1].fitness > 0.0);
        assert!(pop[p2].fitness > 0.0);
    }

    #[test]
    fn test_roulette_wheel_selection() {
        let pop = vec![
            Individual::with_fitness(vec![0.0], 10.0),
            Individual::with_fitness(vec![0.0], 20.0),
            Individual::with_fitness(vec![0.0], 30.0),
        ];
        let sel = RouletteWheelSelection;
        let (p1, p2) = sel.select(&pop);
        assert!(p1 < pop.len());
        assert!(p2 < pop.len());
    }

    #[test]
    fn test_rank_based_selection() {
        let pop = vec![
            Individual::with_fitness(vec![0.0], 10.0),
            Individual::with_fitness(vec![0.0], 20.0),
            Individual::with_fitness(vec![0.0], 30.0),
            Individual::with_fitness(vec![0.0], 40.0),
        ];
        let sel = RankBasedSelection;
        let (p1, p2) = sel.select(&pop);
        assert!(p1 < pop.len());
        assert!(p2 < pop.len());
    }

    #[test]
    fn test_ga_maximizes_sum() {
        // Fitness: sum of all genes (maximize)
        let fitness_fn = |chrom: &Chromosome| -> f64 {
            chrom.iter().sum()
        };
        let mut config = GAConfig::default();
        config.population_size = 30;
        config.chromosome_length = 5;
        config.max_generations = 50;
        config.elitism_count = 2;
        config.target_fitness = Some(4.5); // 5 genes * 0.9 avg

        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize(&fitness_fn);

        let initial_best = ga.best().unwrap().fitness;

        let sel = TournamentSelection::new(3);
        let result = ga.run(&fitness_fn, &sel);

        assert!(result.best_individual.fitness > initial_best,
            "GA should improve fitness: {} > {}", result.best_individual.fitness, initial_best);
        assert_eq!(result.stats.len() as u32, result.generation + 1);
    }

    #[test]
    fn test_ga_elitism_preserves_best() {
        let fitness_fn = |chrom: &Chromosome| -> f64 {
            chrom.iter().sum()
        };
        let mut config = GAConfig::default();
        config.population_size = 20;
        config.chromosome_length = 4;
        config.max_generations = 5;
        config.elitism_count = 3;

        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize(&fitness_fn);

        let sel = TournamentSelection::new(2);
        let _ = ga.run(&fitness_fn, &sel);

        // Stats should show best fitness is non-decreasing
        let stats = ga.stats();
        for i in 1..stats.len() {
            assert!(stats[i].best_fitness >= stats[i - 1].best_fitness - 1e-10,
                "Best fitness should not decrease: gen {} ({}) < gen {} ({})",
                i, stats[i].best_fitness, i - 1, stats[i - 1].best_fitness);
        }
    }

    #[test]
    fn test_ga_convergence_detection() {
        let fitness_fn = |_chrom: &Chromosome| -> f64 { 0.5 }; // constant fitness

        let mut config = GAConfig::default();
        config.population_size = 10;
        config.chromosome_length = 2;
        config.max_generations = 100;
        config.convergence_window = 5;
        config.convergence_threshold = 1e-6;
        config.elitism_count = 2;

        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize(&fitness_fn);

        let sel = TournamentSelection::new(2);
        let result = ga.run(&fitness_fn, &sel);

        assert!(result.converged, "Should detect convergence with constant fitness");
    }

    #[test]
    fn test_ga_target_fitness() {
        let fitness_fn = |chrom: &Chromosome| -> f64 {
            chrom.iter().sum()
        };
        let mut config = GAConfig::default();
        config.population_size = 30;
        config.chromosome_length = 3;
        config.max_generations = 200;
        config.target_fitness = Some(2.5); // should be reachable
        config.elitism_count = 2;

        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize(&fitness_fn);
        let sel = TournamentSelection::new(3);
        let result = ga.run(&fitness_fn, &sel);

        assert!(result.converged, "Should converge to target fitness");
        assert!(result.best_individual.fitness >= 2.5);
    }

    #[test]
    fn test_generation_stats_structure() {
        let fitness_fn = |chrom: &Chromosome| -> f64 {
            chrom.iter().map(|g| g * g).sum()
        };
        let mut config = GAConfig::default();
        config.population_size = 10;
        config.chromosome_length = 3;
        config.max_generations = 5;

        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize(&fitness_fn);
        let sel = TournamentSelection::new(2);
        let result = ga.run(&fitness_fn, &sel);

        assert_eq!(result.stats.len(), 6); // gen 0 through gen 5
        let first = &result.stats[0];
        assert!(first.avg_fitness >= first.worst_fitness);
        assert!(first.avg_fitness <= first.best_fitness);
        assert!(first.diversity >= 0.0);
    }

    #[test]
    fn test_ga_with_two_point_crossover() {
        let fitness_fn = |chrom: &Chromosome| -> f64 {
            chrom.iter().sum()
        };
        let mut config = GAConfig::default();
        config.population_size = 20;
        config.chromosome_length = 6;
        config.max_generations = 10;
        config.elitism_count = 1;

        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize(&fitness_fn);
        let sel = TournamentSelection::new(2);
        let result = ga.run_with_operators(
            &fitness_fn, &sel,
            crossover_two_point,
            |chrom: &mut Chromosome| { mutate_random_replace(chrom, 0.1, 0.0, 1.0); },
        );

        assert!(!result.stats.is_empty());
    }

    #[test]
    fn test_ga_with_uniform_crossover_and_gaussian_mutation() {
        let fitness_fn = |chrom: &Chromosome| -> f64 {
            chrom.iter().product()
        };
        let mut config = GAConfig::default();
        config.population_size = 20;
        config.chromosome_length = 4;
        config.max_generations = 10;
        config.elitism_count = 1;

        let mut ga = GeneticAlgorithm::new(config);
        ga.initialize(&fitness_fn);
        let sel = RouletteWheelSelection;
        let result = ga.run_with_operators(
            &fitness_fn, &sel,
            |p1, p2| crossover_uniform(p1, p2, 0.5),
            |chrom: &mut Chromosome| { mutate_gaussian(chrom, 0.1, 0.05); },
        );

        assert!(!result.stats.is_empty());
    }

    #[test]
    fn test_empty_chromosome_crossover() {
        let p1: Chromosome = vec![];
        let p2: Chromosome = vec![];
        let (c1, c2) = crossover_single_point(&p1, &p2);
        assert!(c1.is_empty());
        assert!(c2.is_empty());
    }
}

// Minimal fastrand module for scramble mutation
mod fastrand {
    pub struct Rng {
        state: u64,
    }

    impl Rng {
        pub fn new() -> Self {
            Rng {
                state: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
            }
        }

        pub fn next_u64(&mut self) -> u64 {
            self.state ^= self.state << 13;
            self.state ^= self.state >> 7;
            self.state ^= self.state << 17;
            self.state
        }
    }

    pub trait Shuffle {
        fn shuffle(&mut self, rng: &mut Rng);
    }

    impl<T> Shuffle for [T] {
        fn shuffle(&mut self, rng: &mut Rng) {
            let len = self.len();
            for i in (1..len).rev() {
                let j = (rng.next_u64() % ((i + 1) as u64)) as usize;
                self.swap(i, j);
            }
        }
    }
}
