"""
SHL Scraper Module

This module provides functionality for scraping assessment data from SHL's product catalog.
"""

from .shl_catalog_scraper import SHLCatalogScraper, Assessment, DataValidator

__all__ = ['SHLCatalogScraper', 'Assessment', 'DataValidator']