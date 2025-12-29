import {
    Box,
    Typography,
    Divider,
    Stack,
    Link,
} from "@mui/material";

import { about } from "../../data/data";

export default function About() {
    return (
        <Box>
            <Typography variant="h4" fontWeight={800} gutterBottom>
                About Me
            </Typography>

            <Stack direction="row" spacing={3} alignItems="flex-start">
                <Box
                    component="img"
                    src={about.photo}
                    alt={about.name}
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
                        {about.blurb}
                    </Typography>

                    <Stack direction="row" spacing={1} mt={1}>
                        {about.linksList.map((l) => (
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

            <Divider sx={{ my: 3 }} />

            <Typography
                variant="h5"
                fontWeight={700}
                color="primary"
                gutterBottom
            >
                Bio
            </Typography>

            <Typography sx={{ maxWidth: 760 }}>
                {about.bio}
            </Typography>
        </Box>
    );
}