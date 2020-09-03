use std::collections::HashMap;
use std::ops::Index;

use rust_algorithms::inference::discrete::NodePotential;
use rust_algorithms::inference::family_tree::make_family_tree;
use rust_algorithms::inference::ising::IsingNode;

pub fn family_tree_example() {
    let alpha = 2.0;
    let (graph, names, name_to_idx) = make_family_tree(alpha);
    let observations: HashMap<usize, usize> = vec![
        (name_to_idx["Lisa"], 1usize),
        (name_to_idx["Bart"], 0),
        (name_to_idx["Maggie"], 0),
        (name_to_idx["Frank"], 1),
        (name_to_idx["Zeke"], 1),
    ]
    .into_iter()
    .collect();
    let (graph, new_to_old) = graph.condition(observations);
    let new_names: Vec<&String> = new_to_old.iter().map(|&i| names.index(i)).collect();
    let node_marginals = graph.compute_marginals().unwrap();
    let marginals = node_marginals.iter().map(|np| np.to_distribution());
    for (&name, marginal) in new_names.iter().zip(marginals) {
        println!("{}", name);
        println!("{}", marginal[1]);
    }
}

pub fn fuzzy_family_tree_example() {
    let alpha = 2.0;
    let beta = 0.8;
    let (mut graph, names, name_to_idx) = make_family_tree(alpha);
    let observations = vec![
        (name_to_idx["Lisa"], 1usize),
        (name_to_idx["Bart"], 0),
        (name_to_idx["Maggie"], 0),
        (name_to_idx["Frank"], 1),
        (name_to_idx["Zeke"], 1),
    ];
    for (idx, obs) in observations {
        graph.set_node_potential(
            idx,
            IsingNode::with_positive_prob(if obs == 1 { beta } else { 1.0 - beta }).unwrap(),
        );
    }
    let node_marginals = graph.compute_marginals().unwrap();
    let marginals = node_marginals.iter().map(|np| np.to_distribution());
    for (name, marginal) in names.iter().zip(marginals) {
        println!("{}", name);
        println!("{}", marginal[1]);
    }
}

pub fn main() {
    // family_tree_example();
    fuzzy_family_tree_example();
}
