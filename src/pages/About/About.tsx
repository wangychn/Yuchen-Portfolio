import {
    Box,
    Typography,
    Divider,
    Stack,
    Link,
} from "@mui/material";

import { paragraphs, personal_info } from "../../data/data";

export default function About() {
    return (
        <Box>
            <Typography variant="h4" fontWeight={800} gutterBottom>
                About Me
            </Typography>

            <Stack direction="row" spacing={3} alignItems="flex-start">
                <Box
                    component="img"
                    src={personal_info.photo}
                    alt={personal_info.name}
                    sx={{
                        width: 190,
                        height: 250,
                        borderRadius: 3,
                        objectFit: "cover",
                    }}
                />

                <Box>
                    <Typography sx={{
                        maxWidth: 640

                    }}>
                        {paragraphs.blurb}
                    </Typography>

                    <Stack direction="row" spacing={1} mt={1}>
                        {personal_info.linksList.map((l) => (
                            <Link
                                key={l.href}
                                href={l.href}
                                target="_blank"
                                underline="hover"
                            >
                                {l.label}
                            </Link>
                        ))}
                    </Stack>
                </Box>
            </Stack>

            <Divider sx={{ my: 5 }} />

            <Typography
                variant="h5"
                fontWeight={700}
                color="primary"
                gutterBottom
            >
                Bio
            </Typography>

            <Typography sx={{ maxWidth: 760 }}>
                {paragraphs.bio}
            </Typography>

            <Divider sx={{ my: 5 }} />

            <Typography
                variant="h5"
                fontWeight={700}
                color="primary"
                gutterBottom
            >
                Currently working on
            </Typography>

            <Typography sx={{ maxWidth: 760 }}>
                {paragraphs.current_projects}
            </Typography>


        </Box>
    );
}