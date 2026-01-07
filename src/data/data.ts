import photo from "../assets/yuchen_headshot2.jpg";
import profile_photo from "../assets/identicon.png";

export const personal_info = {
    name: "Yuchen Wang",
    title: "Student @ University of Michigan",
    photo,
    profile_photo,
    linksList: [
        { label: "GitHub", href: "https://github.com/wangychn" },
        { label: "LinkedIn", href: "https://linkedin.com/in/yuchen-daniel-wang/" },
        { label: "Email", href: "wangyuch@umich.edu" },
        { label: "Calendar", href: "https://calendar.app.google/kdphBnZfTj7cBLdd8" },
    ],
};

export const paragraphs = {
    blurb: (
        "I’m a Computer Science at the University of Michigan my focus is "
        + "distributed training of models and exploring novel model architectures."
    ),
    bio: (
        "I am a Computer Science student at the University of Michigan with interests in systems, "
        + "machine learning, and applied AI research. I have worked on projects spanning reinforcement learning, "
        + "large language models, and full-stack systems, and have experience deploying production-level software. "
        + "I had a great time working as a software engineer at Amazon Web Services, but  "
        + "want to see how far I can go with ai research  "
        + "My current focus is on developing robust, distrbuted AI systems. "
    ),
    current_projects: [
        {
            title: "In-browser RL demos (ONNX + WebAssembly)",
            details: [
                "Shipping interactive CartPole/DQN demos that run fully in the browser.",
                "Improving rendering quality (hi-res canvas) and responsive UI behavior.",
                "Understanding runtime constraints: WASM loading, threading, and model sizes.",
            ],
        },
        {
            title: "Distributed GPU Training",
            details: [
                "Studying model export formats, runtimes, and inference trade-offs.",
                "Building intuition for performance bottlenecks and profiling workflows.",
                "https://huggingface.co/spaces/weege007/ultrascale-playbook?section=data_parallelism",
            ],
        },
        {
            title: "Grad school + research prep!",
            details: [
                "Organizing research-style writeups, slides, and experiments for applications.",
                "Iterating on project narratives and technical depth (systems + ML).",
            ],
        },
    ] as const,
};