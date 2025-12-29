import styles from "./Sidebar.module.css";

type Props = {
    current: string;
    onChange: (page: string) => void;
}

const sidebarItems: { key: Page; label: string }[] = [
    { key: "about", label: "About" },
    { key: "experience", label: "Experience" },
    { key: "projects", label: "Projects" },
];

export default function Sidebar({ current, onChange }: Props) {
    return (
        <aside className={styles.sidebar}>
            <div className={styles.inner}>
                {/* Profile block */}
                <div className={styles.profile}>
                    <div className={styles.avatar} aria-hidden="true" />

                    <div className={styles.name}>Yuchen Wang</div>
                    <div className={styles.subtitle}>Student @ University of Michigan</div>

                    {/* Icon row (swap to Ant icons later) */}
                    <div className={styles.iconRow} aria-label="Social links">
                        <a className={styles.iconBtn} href="#" title="Email" aria-label="Email">
                            ✉️
                        </a>
                        <a className={styles.iconBtn} href="#" title="GitHub" aria-label="GitHub">
                            🐙
                        </a>
                        <a className={styles.iconBtn} href="#" title="LinkedIn" aria-label="LinkedIn">
                            in
                        </a>
                        <a className={styles.iconBtn} href="#" title="Calendar" aria-label="Calendar">
                            📅
                        </a>
                        <a className={styles.iconBtn} href="#" title="X/Twitter" aria-label="X/Twitter">
                            x
                        </a>
                    </div>
                </div>

                {/* Nav */}
                <nav className={styles.nav} aria-label="Primary">
                    {sidebarItems.map((item) => {
                        // const active = item.key === current;

                        return (
                            <button
                                key={item.key}
                                className={`${styles.navItem} ${current === item.key ? styles.active : ""}`}
                                onClick={() => onChange(item.key)}
                            >
                                <span>{item.label}</span>
                                <span className={styles.bridge} aria-hidden="true" />
                            </button>
                        );
                    })}
                </nav>

                <div className={styles.footer}>© 2019–2025 Yuchen Wang</div>
            </div>
        </aside>
    );
}