// src/data/projects.ts
export interface Project {
    slug: string;
    title: string;
    description: string;
    demoUrl: string;
}

export const projects: Project[] = [
    {
        slug: 'CartPole-DQN',
        title: '1. CartPole-DQN',
        description: `This project is my implementation of the 
                        classic reinforcement learning Cart-Pole 
                        problem. \n \n It is also the source code of my demo above!`,
        demoUrl: '/projects/Cartpole'
    },
    {
        slug: 'MNIST-VAE',
        title: 'Variational Auto Encoder',
        description: `This project contains my PyTorch implementation of 
                    a Variational Autoencoder (VAE) for the MNIST dataset. 
                    \n \n This is among my first steps into GenAI.`,
        demoUrl: '/projects/game-of-life'
    },
    // …more entries…
];


