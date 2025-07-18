% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/client.R
\name{pjrt_client}
\alias{pjrt_client}
\title{Create a Client}
\usage{
pjrt_client(platform)
}
\arguments{
\item{platform}{(\code{character(1)})\cr
Platform name (e.g., "cpu", "cuda", "metal").}
}
\value{
\code{PJRTClient}
}
\description{
Create a PJRT client for a specific device.
}
