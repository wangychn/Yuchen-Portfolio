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
        "I’m a Computer Science at the University of Michigan. My interests are in ML systems for training and inference. " +
        "My goal is to build infrastructure that expands what kinds of models can be trained and deployed."
    ),
    bio: (
        "My research direction lie in designing the training and inference systems that make modern AI models possible, including distributed execution, "
        + "runtime scheduling, and efficient use of heterogeneous hardware. \n\n"
        + "I have worked on projects spanning reinforcement learning, large language models, and systems engineering, and have experience building "
        + "production-level software during my time at Amazon Web Services. \n\n"
        + "My goal is to develop novel ML systems that expand the scale and capabilities of future AI models."
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