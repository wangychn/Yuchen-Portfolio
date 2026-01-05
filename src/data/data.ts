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
        "I’m a Computer Science and Business student at the University of Michigan who enjoys building "
        + "reliable systems and exploring how machine learning ideas translate into real, usable software."
    ),
    bio: (
        "I am a Computer Science and Business student at the University of Michigan with interests in systems, "
        + "machine learning, and applied AI research. I have worked on projects spanning reinforcement learning, "
        + "large language models, and full-stack systems, and have experience deploying production-level software "
        + "through internships and research collaborations. Previously, I interned at a startup accelerator and "
        + "worked on applied software and data projects across multiple early-stage companies. I have also served "
        + "as a teaching assistant and peer mentor in technical courses, and I actively build research-oriented "
        + "projects that bridge theory and real-world deployment. My current focus is on developing robust, "
        + "interpretable AI systems and exploring research directions at the intersection of learning, reasoning, "
        + "and systems."
    ),
    current_projects: (
        "Building and deploying reinforcement learning demos (e.g., Cart-Pole DQN) with ONNX and WebAssembly for in-browser inference."
        + "Exploring AI systems and infrastructure topics, including model export, runtimes, and deployment trade-offs."
        + "Developing full-stack and cloud-backed applications that integrate modern ML workflows."
        + "Preparing research-oriented projects and materials for graduate school and industry applications."
    )
};