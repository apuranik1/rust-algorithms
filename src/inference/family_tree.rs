use std::collections::BTreeMap;

use crate::inference::discrete::DiscreteUndirectedGraph;
use crate::inference::ising::{IsingEdge, IsingNode};

pub fn make_family_tree(
    alpha: f64,
) -> (
    DiscreteUndirectedGraph<IsingNode, IsingEdge>,
    Vec<String>,
    BTreeMap<String, usize>,
) {
    assert!(alpha > 0.0);
    let p_aligned = alpha / (1.0 + alpha);
    let nodes: Vec<String> = vec![
        "Orville", "Abraham", "Hubert", "Homer", "Tyrone", "Cyrus", "Lisa", "Bart", "Maggie",
        "Frank", "Zeke",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let name_to_idx: BTreeMap<String, usize> = nodes
        .iter()
        .enumerate()
        .map(|(a, b)| (b.clone(), a))
        .collect();
    let edges: Vec<(usize, usize)> = {
        let named_edges = vec![
            ("Lisa", "Homer"),
            ("Bart", "Homer"),
            ("Maggie", "Homer"),
            ("Frank", "Tyrone"),
            ("Zeke", "Tyrone"),
            ("Homer", "Abraham"),
            ("Tyrone", "Hubert"),
            ("Cyrus", "Hubert"),
            ("Abraham", "Orville"),
            ("Hubert", "Orville"),
        ];
        named_edges
            .iter()
            .map(|(name1, name2)| (name_to_idx[*name1], name_to_idx[*name2]))
            .collect()
    };
    let node_potentials = (0..nodes.len()).map(|_| IsingNode::new(0.0)).collect();
    let family_edge = IsingEdge::with_aligned_prob(p_aligned).unwrap();
    let edge_potentials: BTreeMap<(usize, usize), IsingEdge> = edges
        .into_iter()
        .map(|edge| (edge, family_edge.clone()))
        .collect();
    (
        DiscreteUndirectedGraph::new(node_potentials, edge_potentials),
        nodes,
        name_to_idx,
    )
}
